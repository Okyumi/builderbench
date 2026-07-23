"""Focused tests for the DCC data-only checkpoint schema.

These exercise the checkpoint boundary that used to crash
``continual_crl_dcc.py`` (``AttributeError: Can't pickle local object
'chain.<locals>.init_fn'``) without needing MuJoCo / the BuilderBench env:
everything runs on tiny Flax/Optax pytrees.

Coverage:
  * data-only roundtrip (and proof the raw TrainState is unpicklable);
  * TrainState reconstruction with a fresh module/optimiser;
  * atomic save behaviour (no temp litter; failed write leaves dest intact);
  * corrupt / incomplete / unsupported / mismatched checkpoint rejection;
  * auto_resume stopping at the last valid contiguous task.

Run from anywhere:  pytest rl/impls/tests/test_dcc_checkpoint.py
"""
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
import flax.linen as nn

# Make ``utils`` importable regardless of the working directory pytest runs in.
_IMPLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IMPLS_DIR not in sys.path:
    sys.path.insert(0, _IMPLS_DIR)

from utils import dcc_checkpoint as dcc  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny fixtures (mirror the Actor + decomposed critic groups, in miniature)
# ---------------------------------------------------------------------------


class TinyActor(nn.Module):
    """Two-input Flax module, shaped like the real Actor(s, g_repr)."""

    @nn.compact
    def __call__(self, s, g):
        x = jnp.concatenate([s, g], axis=-1)
        x = nn.Dense(8)(x)
        return nn.Dense(4)(x)


def _make_actor_state(seed=0, lr=1e-3):
    actor = TinyActor()
    params = actor.init(jax.random.PRNGKey(seed),
                        jnp.ones((1, 3)), jnp.ones((1, 2)))
    tx = optax.adam(lr)
    state = TrainState.create(apply_fn=actor.apply, params=params, tx=tx)
    return actor, tx, state


def _make_groups(lr=1e-3):
    """Realistic ``{name: (params, opt_state)}`` including two *unbuilt*
    optional goal groups stored as ``(None, None)`` (as the real driver does
    for psi_task / psi_proj in the 'shared'/'add' mode)."""
    tx = optax.adam(lr)
    p1 = {'w': jnp.ones((2, 2)), 'b': jnp.zeros((2,))}
    p2 = {'w': jnp.full((3,), 2.0, dtype=jnp.float32)}
    return {
        'b_shared': (p1, tx.init(p1)),
        'h_phi': (p2, tx.init(p2)),
        'h_dyn': (p2, tx.init(p2)),
        'phi_task': (p1, tx.init(p1)),
        'psi_shared': (p2, tx.init(p2)),
        'psi_task': (None, None),
        'psi_proj': (None, None),
    }


def _leaves_allclose(a, b):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    return all(np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _valid_payload(task_idx=0, config=None):
    _, _, state = _make_actor_state()
    return dcc.build_payload(
        task_idx=task_idx, task_id=f'cube-1-task{task_idx + 1}',
        actor_state=state, critic_groups=_make_groups(),
        config=config or {},
    )


# ---------------------------------------------------------------------------
# 1. Data-only roundtrip
# ---------------------------------------------------------------------------


def test_raw_trainstate_is_unpicklable_but_payload_is_safe():
    """Reproduces the historical failure and proves the payload avoids it."""
    _, _, state = _make_actor_state()
    with pytest.raises(Exception) as exc:
        pickle.dumps(state)
    # The optax.chain closure is the culprit either on the dump or load side.
    assert 'init_fn' in str(exc.value) or 'local object' in str(exc.value)

    payload = dcc.build_payload(
        task_idx=0, task_id='t', actor_state=state,
        critic_groups=_make_groups(), config={})
    # Must serialise cleanly — no tx / apply_fn closures in the payload.
    blob = pickle.dumps(payload)
    assert len(blob) > 0


def test_data_only_roundtrip(tmp_path):
    payload = _valid_payload(task_idx=0)
    path = tmp_path / 'task_00.pkl'
    dcc.save_payload(path, payload)

    loaded = dcc.load_payload(path, expected_task_idx=0)
    assert loaded is not None
    assert loaded['format'] == dcc.CKPT_FORMAT
    assert loaded['version'] == dcc.CKPT_VERSION
    assert loaded['task_idx'] == 0
    assert _leaves_allclose(loaded['actor']['params'],
                            payload['actor']['params'])
    assert set(loaded['critic_groups']) == set(dcc.CRITIC_GROUP_NAMES)


# ---------------------------------------------------------------------------
# 2. TrainState reconstruction
# ---------------------------------------------------------------------------


def test_trainstate_reconstruction(tmp_path):
    actor, _, state = _make_actor_state()
    # Take a real optimiser step so step > 0 and opt_state is non-trivial.
    grads = jax.tree_util.tree_map(jnp.ones_like, state.params)
    state = state.apply_gradients(grads=grads)

    payload = dcc.build_payload(
        task_idx=0, task_id='t', actor_state=state,
        critic_groups=_make_groups(), config={})
    path = tmp_path / 'task_00.pkl'
    dcc.save_payload(path, payload)
    loaded = dcc.load_payload(path, expected_task_idx=0)

    # Reconstruct with a FRESH module + optimiser (the closures are rebuilt).
    restored = dcc.restore_actor_state(loaded, actor, optax.adam(1e-3))
    assert _leaves_allclose(restored.params, state.params)
    assert _leaves_allclose(restored.opt_state, state.opt_state)
    assert int(restored.step) == int(state.step)

    # The reconstructed state must be functional (opt_state/tx wired up).
    stepped = restored.apply_gradients(
        grads=jax.tree_util.tree_map(jnp.ones_like, restored.params))
    assert int(stepped.step) == int(state.step) + 1


def test_critic_groups_reconstruction(tmp_path):
    groups = _make_groups()
    payload = dcc.build_payload(
        task_idx=0, task_id='t', actor_state=_make_actor_state()[2],
        critic_groups=groups, config={})
    path = tmp_path / 'task_00.pkl'
    dcc.save_payload(path, payload)
    loaded = dcc.load_payload(path)

    restored = dcc.restore_critic_groups(loaded)
    assert set(restored) == set(dcc.CRITIC_GROUP_NAMES)
    # Unbuilt optional groups come back as (None, None).
    assert restored['psi_task'] == (None, None)
    assert restored['psi_proj'] == (None, None)
    # Present groups keep their values and optimiser state.
    assert _leaves_allclose(restored['b_shared'][0], groups['b_shared'][0])
    assert _leaves_allclose(restored['b_shared'][1], groups['b_shared'][1])


# ---------------------------------------------------------------------------
# 3. Atomic save behaviour
# ---------------------------------------------------------------------------


def test_atomic_save_leaves_no_tempfiles(tmp_path):
    path = tmp_path / 'sub' / 'task_00.pkl'
    dcc.save_payload(path, _valid_payload())
    assert path.exists()
    # No .tmp_ckpt_* litter in the target directory.
    assert not list(path.parent.glob('.tmp_ckpt_*'))


def test_failed_write_leaves_existing_checkpoint_intact(tmp_path):
    path = tmp_path / 'task_00.pkl'
    dcc.atomic_pickle_dump(path, {'ok': np.arange(4)})
    original = path.read_bytes()

    # A serialisation failure mid-write must not touch the good file...
    with pytest.raises(Exception):
        dcc.atomic_pickle_dump(path, {'bad': lambda x: x})  # lambda unpicklable
    assert path.read_bytes() == original
    # ...and must not leave a temp file behind.
    assert not list(tmp_path.glob('.tmp_ckpt_*'))


def test_failed_first_write_creates_no_checkpoint_and_autoresume_reruns(tmp_path):
    """A failed task-0 write must NOT be seen as a completed task."""
    path = tmp_path / 'task_00.pkl'
    with pytest.raises(Exception):
        dcc.atomic_pickle_dump(path, {'bad': lambda x: x})
    assert not path.exists()
    last = dcc.last_valid_contiguous_task(
        lambda i: tmp_path / f'task_{i:02d}.pkl', num_tasks=3)
    assert last == -1  # task 0 will be re-run, not skipped


# ---------------------------------------------------------------------------
# 4. Corrupt / incomplete / unsupported / mismatched rejection
# ---------------------------------------------------------------------------


def test_reject_zero_byte(tmp_path):
    path = tmp_path / 'task_00.pkl'
    path.write_bytes(b'')
    with pytest.raises(dcc.CheckpointError):
        dcc.load_payload(path)
    assert dcc.is_valid_ckpt(path) is False


def test_reject_garbage_bytes(tmp_path):
    path = tmp_path / 'task_00.pkl'
    path.write_bytes(b'not a pickle at all \x00\x01\x02')
    with pytest.raises(dcc.CheckpointError):
        dcc.load_payload(path)
    assert dcc.is_valid_ckpt(path) is False


def test_reject_wrong_format(tmp_path):
    path = tmp_path / 'task_00.pkl'
    dcc.atomic_pickle_dump(path, {'hello': 'world'})
    with pytest.raises(dcc.CheckpointError):
        dcc.load_payload(path)


def test_reject_unsupported_version(tmp_path):
    payload = _valid_payload()
    payload['version'] = 999
    path = tmp_path / 'task_00.pkl'
    dcc.atomic_pickle_dump(path, payload)
    with pytest.raises(dcc.CheckpointError):
        dcc.load_payload(path)


def test_reject_incomplete_payload(tmp_path):
    payload = _valid_payload()
    del payload['actor']
    path = tmp_path / 'task_00.pkl'
    dcc.atomic_pickle_dump(path, payload)
    with pytest.raises(dcc.CheckpointError):
        dcc.load_payload(path)


def test_reject_task_idx_mismatch(tmp_path):
    payload = _valid_payload(task_idx=0)
    path = tmp_path / 'task_00.pkl'
    dcc.save_payload(path, payload)
    with pytest.raises(dcc.CheckpointError):
        dcc.load_payload(path, expected_task_idx=1)


def test_reject_legacy_full_trainstate_with_actionable_error(tmp_path):
    """A legacy dict (no versioned header) must be rejected clearly, never
    silently mis-resumed."""
    path = tmp_path / 'task_00.pkl'
    dcc.atomic_pickle_dump(path, {'actor_state': {'p': 1},
                                  'critic_groups': {}})
    with pytest.raises(dcc.CheckpointError) as exc:
        dcc.load_payload(path)
    assert 'legacy' in str(exc.value).lower()


def test_reject_structural_config_mismatch(tmp_path):
    payload = _valid_payload(config={'combine_mode': 'add',
                                     'goal_encoder_mode': 'shared',
                                     'rep_size': 64})
    path = tmp_path / 'task_00.pkl'
    dcc.save_payload(path, payload)
    with pytest.raises(dcc.CheckpointError):
        dcc.load_payload(path, expected_config={'combine_mode': 'concat',
                                                'goal_encoder_mode': 'shared',
                                                'rep_size': 64})
    # Same structural config still loads.
    assert dcc.load_payload(path, expected_config={'combine_mode': 'add',
                                                   'goal_encoder_mode': 'shared',
                                                   'rep_size': 64}) is not None


# ---------------------------------------------------------------------------
# 5. auto_resume: last valid contiguous task
# ---------------------------------------------------------------------------


def _write_tasks(tmp_path, indices):
    for i in indices:
        dcc.save_payload(tmp_path / f'task_{i:02d}.pkl', _valid_payload(task_idx=i))


def _scan(tmp_path, num_tasks):
    return dcc.last_valid_contiguous_task(
        lambda i: tmp_path / f'task_{i:02d}.pkl', num_tasks=num_tasks)


def test_autoresume_none(tmp_path):
    assert _scan(tmp_path, 3) == -1


def test_autoresume_all_valid(tmp_path):
    _write_tasks(tmp_path, [0, 1, 2])
    assert _scan(tmp_path, 3) == 2


def test_autoresume_stops_at_gap(tmp_path):
    # Tasks 0 and 2 valid, task 1 missing -> resume from task 1 (last=0).
    _write_tasks(tmp_path, [0, 2])
    assert _scan(tmp_path, 3) == 0


def test_autoresume_stops_at_corrupt(tmp_path):
    _write_tasks(tmp_path, [0, 1, 2])
    # Corrupt task 1 (mid-sequence): last valid contiguous is task 0.
    (tmp_path / 'task_01.pkl').write_bytes(b'')
    assert _scan(tmp_path, 3) == 0


def test_autoresume_zero_byte_task0_is_not_complete(tmp_path):
    """The exact historical failure mode: a zero-byte task-0 checkpoint left
    by a crashed run must NOT make the next run skip task 0."""
    (tmp_path / 'task_00.pkl').write_bytes(b'')
    assert _scan(tmp_path, 3) == -1
