"""Versioned, data-only checkpointing for the DCC continual-CRL driver.

Historical failure
------------------
``continual_crl_dcc.py`` used to pickle the whole actor ``TrainState``::

    save_ckpt(..., dict(actor_state=actor_state, critic_groups=groups))

A Flax ``TrainState`` stores two objects that ``pickle`` cannot serialise:

  * ``tx``  — the Optax ``GradientTransformation``. ``optax.adam`` is built
    with ``optax.chain(...)``, whose ``init_fn`` is a *local* closure, so
    pickling raises ``AttributeError: Can't pickle local object
    'chain.<locals>.init_fn'``.
  * ``apply_fn`` — a bound method of the Flax module.

Every DCC ablation therefore crashed at the first task boundary (right
after task 0 finished training), leaving the sweep unable to progress.

Fix
---
Persist a **data-only** payload — actor ``step`` / ``params`` /
``opt_state`` and each critic group's ``params`` / ``opt_state`` — and
reconstruct the ``TrainState`` after loading, using the *current* Actor
module and optimiser. The optimiser/apply closures are rebuilt at load
time and never touch the on-disk format. This mirrors what the baseline
``continual_crl.py`` already does (it saves ``actor_params`` /
``critic_params`` only).

The module is intentionally free of any environment/MuJoCo imports so the
checkpoint boundary can be unit-tested with tiny Flax/Optax pytrees.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

CKPT_FORMAT = 'builderbench-dcc-ckpt'
# v1 == legacy full-``TrainState`` payload. It never wrote successfully
# (see module docstring) so no valid v1 file can exist on disk; it is listed
# only so the loader can name it in the error message.
CKPT_VERSION = 2
SUPPORTED_VERSIONS = (2,)

# Canonical order of the DCC critic parameter groups. Optional groups
# (``psi_task`` / ``psi_proj``) are present for every goal mode but may carry
# ``None`` params when the corresponding module is not built.
CRITIC_GROUP_NAMES: Tuple[str, ...] = (
    'b_shared', 'h_phi', 'h_dyn', 'phi_task',
    'psi_shared', 'psi_task', 'psi_proj',
)

# Config keys whose value changes the parameter-tree *shape*. A mismatch on
# any of these means the on-disk params cannot be reconstructed with the
# current networks, so the loader refuses rather than reshaping silently.
STRUCTURAL_CONFIG_KEYS: Tuple[str, ...] = (
    'combine_mode', 'goal_encoder_mode', 'rep_size',
    'phi_task_width', 'phi_task_depth',
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is corrupt, incomplete, or unsupported."""


# ---------------------------------------------------------------------------
# Array (de)serialisation helpers
# ---------------------------------------------------------------------------


def _to_numpy(tree: Any) -> Any:
    """Map every JAX/NumPy array leaf to a host ``np.ndarray``.

    Non-array leaves (python scalars, ``None`` subtrees) pass through so the
    pytree structure is preserved exactly.
    """
    return jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, (jnp.ndarray, np.ndarray)) else x,
        tree,
    )


def _to_jax(tree: Any) -> Any:
    """Inverse of :func:`_to_numpy`: NumPy array leaves become ``jnp`` arrays."""
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
        tree,
    )


# ---------------------------------------------------------------------------
# Payload construction / validation
# ---------------------------------------------------------------------------


def build_payload(
    *,
    task_idx: int,
    task_id: str,
    actor_state: TrainState,
    critic_groups: Mapping[str, Tuple[Any, Any]],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a versioned, pickle-safe, data-only checkpoint payload.

    Args:
      task_idx: index of the task that just completed.
      task_id: human-readable task id (e.g. ``'cube-1-task1'``).
      actor_state: Flax ``TrainState``; only ``step`` / ``params`` /
        ``opt_state`` are kept — never ``apply_fn`` or ``tx``.
      critic_groups: ``{name: (params, opt_state)}``. ``None`` params are
        stored as-is (optional goal-encoder groups).
      config: small structural fingerprint used for validation on load.
    """
    actor = {
        'step': _to_numpy(actor_state.step),
        'params': _to_numpy(actor_state.params),
        'opt_state': _to_numpy(actor_state.opt_state),
    }

    groups: Dict[str, Dict[str, Any]] = {}
    for name in CRITIC_GROUP_NAMES:
        if name not in critic_groups:
            continue
        params, opt_state = critic_groups[name]
        groups[name] = {
            'params': _to_numpy(params),
            'opt_state': _to_numpy(opt_state),
        }

    return {
        'format': CKPT_FORMAT,
        'version': CKPT_VERSION,
        'task_idx': int(task_idx),
        'task_id': task_id,
        'actor': actor,
        'critic_groups': groups,
        'config': dict(config) if config is not None else {},
    }


def _check_config(stored: Mapping[str, Any],
                  expected: Mapping[str, Any],
                  where: str) -> None:
    for key in STRUCTURAL_CONFIG_KEYS:
        if key in expected and key in stored and stored[key] != expected[key]:
            raise CheckpointError(
                f'{where}: config mismatch on {key!r} '
                f'(checkpoint={stored[key]!r}, run={expected[key]!r}). '
                f'This checkpoint was written for a different network shape; '
                f'point --checkpoint_dir at the matching run or delete it.')


def validate_payload(
    obj: Any,
    where: str = '<payload>',
    expected_task_idx: Optional[int] = None,
    expected_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate an in-memory payload, raising :class:`CheckpointError`.

    Checks: dict type, recognised ``format``/``version``, presence of the
    required actor/critic keys, and (optionally) the task index and the
    structural config fingerprint. Returns the object on success.
    """
    if not isinstance(obj, dict):
        raise CheckpointError(
            f'{where}: expected a dict payload, got {type(obj).__name__}.')

    fmt = obj.get('format')
    if fmt != CKPT_FORMAT:
        # Legacy full-``TrainState`` payloads had no versioned header. They
        # could never be written (pickling the Optax tx fails), so any such
        # file is unusable; say so instead of silently mis-resuming.
        if 'format' not in obj and ('actor_state' in obj or 'critic_groups' in obj):
            raise CheckpointError(
                f'{where}: looks like a legacy full-TrainState checkpoint with '
                f'no versioned schema. Legacy DCC checkpoints could never be '
                f'written successfully (pickling the Optax optimiser fails), so '
                f'this file is unusable. Delete the checkpoint directory and '
                f're-run the affected task. See '
                f'doc/2026-07-23_dcc_task_boundary_fix.md.')
        raise CheckpointError(
            f'{where}: unrecognised checkpoint format {fmt!r} '
            f'(expected {CKPT_FORMAT!r}).')

    ver = obj.get('version')
    if ver not in SUPPORTED_VERSIONS:
        raise CheckpointError(
            f'{where}: unsupported checkpoint version {ver!r} '
            f'(supported: {SUPPORTED_VERSIONS}).')

    for key in ('task_idx', 'task_id', 'actor', 'critic_groups'):
        if key not in obj:
            raise CheckpointError(f'{where}: missing required key {key!r}.')

    actor = obj['actor']
    if not isinstance(actor, Mapping):
        raise CheckpointError(f'{where}: "actor" must be a mapping.')
    for key in ('step', 'params', 'opt_state'):
        if key not in actor:
            raise CheckpointError(f'{where}: actor missing {key!r}.')

    if not isinstance(obj['critic_groups'], Mapping):
        raise CheckpointError(f'{where}: "critic_groups" must be a mapping.')

    if expected_task_idx is not None and int(obj['task_idx']) != int(expected_task_idx):
        raise CheckpointError(
            f'{where}: task_idx mismatch (checkpoint={obj["task_idx"]}, '
            f'expected={expected_task_idx}).')

    if expected_config is not None:
        _check_config(obj.get('config', {}), expected_config, where)

    return obj


# ---------------------------------------------------------------------------
# Atomic pickle IO
# ---------------------------------------------------------------------------


def atomic_pickle_dump(path: os.PathLike | str, obj: Any) -> None:
    """Pickle ``obj`` to ``path`` atomically.

    Writes to a uniquely-named temp file in the *same* directory, flushes and
    ``fsync``s it, then ``os.replace``s it onto ``path`` (atomic on POSIX).
    On any failure the temp file is removed and ``path`` is left untouched —
    so a serialisation failure or preemption can never leave a partially
    written checkpoint that ``auto_resume`` would treat as complete.
    """
    path = os.fspath(path)
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)

    fd, tmp = tempfile.mkstemp(prefix='.tmp_ckpt_', suffix='.pkl', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_payload(path: os.PathLike | str, payload: Mapping[str, Any]) -> None:
    """Validate then atomically persist a checkpoint payload."""
    validate_payload(payload, where=os.fspath(path))
    atomic_pickle_dump(path, payload)


def load_payload(
    path: os.PathLike | str,
    expected_task_idx: Optional[int] = None,
    expected_config: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Load and validate a checkpoint.

    Returns ``None`` if the file does not exist. Raises
    :class:`CheckpointError` for zero-byte, unpicklable, corrupt, incomplete,
    or unsupported checkpoints (and on task/config mismatch).
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        raise CheckpointError(
            f'{path}: zero-byte checkpoint (corrupt or partially written).')
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except CheckpointError:
        raise
    except Exception as e:  # noqa: BLE001 - surface any unpickling failure
        raise CheckpointError(f'{path}: failed to unpickle ({e!r}).') from e
    return validate_payload(obj, path, expected_task_idx, expected_config)


def is_valid_ckpt(
    path: os.PathLike | str,
    expected_task_idx: Optional[int] = None,
    expected_config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return ``True`` iff ``path`` is a loadable, valid checkpoint.

    Never raises: used by ``auto_resume`` to decide whether a task counts as
    complete. Missing, zero-byte, corrupt, or mismatched files return ``False``.
    """
    try:
        return load_payload(path, expected_task_idx, expected_config) is not None
    except CheckpointError:
        return False


def last_valid_contiguous_task(
    path_for_task,
    num_tasks: int,
    expected_config: Optional[Mapping[str, Any]] = None,
    on_invalid=None,
) -> int:
    """Index of the last VALID, CONTIGUOUS completed task (or ``-1``).

    ``path_for_task(idx) -> path``. The scan stops at the first task whose
    checkpoint is missing or fails validation, so a corrupt/partial task
    checkpoint is re-run rather than skipped, and a valid checkpoint sitting
    *after* a gap is ignored. ``on_invalid(idx, path)`` is invoked for an
    existing-but-invalid checkpoint (e.g. for logging).
    """
    last = -1
    for probe in range(num_tasks):
        path = path_for_task(probe)
        if is_valid_ckpt(path, expected_task_idx=probe,
                         expected_config=expected_config):
            last = probe
            continue
        if on_invalid is not None and os.path.exists(os.fspath(path)):
            on_invalid(probe, path)
        break
    return last


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def restore_actor_state(
    payload: Mapping[str, Any],
    actor_module: Any,
    actor_optimizer: Any,
) -> TrainState:
    """Rebuild a Flax ``TrainState`` from a data-only payload.

    The ``apply_fn`` and ``tx`` closures are supplied fresh from the current
    ``actor_module`` / ``actor_optimizer``; only ``step`` / ``params`` /
    ``opt_state`` come from disk.
    """
    actor = payload['actor']
    params = _to_jax(actor['params'])
    opt_state = _to_jax(actor['opt_state'])
    step = _to_jax(actor['step'])

    state = TrainState.create(
        apply_fn=actor_module.apply, params=params, tx=actor_optimizer)
    return state.replace(step=step, opt_state=opt_state)


def restore_critic_groups(
    payload: Mapping[str, Any],
    group_names: Sequence[str] = CRITIC_GROUP_NAMES,
) -> Dict[str, Tuple[Any, Any]]:
    """Return ``{name: (params, opt_state)}`` with JAX arrays.

    Groups absent from the payload (or stored with ``None`` params) come back
    as ``(None, None)`` so the structure matches a freshly-initialised store.
    """
    stored = payload['critic_groups']
    out: Dict[str, Tuple[Any, Any]] = {}
    for name in group_names:
        g = stored.get(name)
        if g is None:
            out[name] = (None, None)
        else:
            out[name] = (_to_jax(g.get('params')), _to_jax(g.get('opt_state')))
    return out
