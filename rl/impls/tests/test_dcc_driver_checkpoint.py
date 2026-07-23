"""Driver-level wiring test for continual_crl_dcc checkpointing.

The DCC driver pulls in the full training stack (tyro, wandb, mujoco, the
BuilderBench env). None of that is needed to exercise the checkpoint
boundary, so this test stubs those modules and then drives the *actual*
``save_ckpt`` / ``load_ckpt`` / ``auto_resume`` / ``_ckpt_config`` wrappers
against the real ``dcc_checkpoint`` implementation, using the driver's own
``Actor`` module.

If the stubbing is insufficient in some environment, the test skips rather
than failing (the substance is covered by ``test_dcc_checkpoint.py``).
"""
import os
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

_IMPLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IMPLS_DIR not in sys.path:
    sys.path.insert(0, _IMPLS_DIR)


def _install_stubs():
    """Register lightweight stand-ins for the heavy imports so the DCC driver
    module body can execute. Returns the list of module names we injected."""
    injected = []

    def stub(name, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        injected.append(name)
        return mod

    if 'tyro' not in sys.modules:
        stub('tyro', cli=lambda *a, **k: None)
    if 'wandb' not in sys.modules:
        stub('wandb', init=lambda *a, **k: None, log=lambda *a, **k: None,
             finish=lambda *a, **k: None)
    if 'wandb_osh' not in sys.modules:
        stub('wandb_osh')
    if 'wandb_osh.hooks' not in sys.modules:
        stub('wandb_osh.hooks', TriggerWandbSyncHook=object)
    if 'mujoco' not in sys.modules:
        # ``utils.wrapper`` evaluates ``mujoco.MjModel`` as a return annotation
        # at import time, and other helpers touch further attributes. Resolve
        # *any* attribute to a throwaway type so the module body can execute.
        mj = stub('mujoco', rollout=types.SimpleNamespace())
        mj.__getattr__ = lambda name: type(name, (), {})
    if 'rl_metrics' not in sys.modules:
        stub('rl_metrics', compute_all_metrics_dcc=lambda *a, **k: {})
    # Prevent the real (mujoco-importing) env package from loading.
    if 'builderbench' not in sys.modules:
        stub('builderbench')
    if 'builderbench.env_utils' not in sys.modules:
        stub('builderbench.env_utils', make_env=lambda *a, **k: (None, None))
    return injected


@pytest.fixture(scope='module')
def driver():
    injected = _install_stubs()
    try:
        import continual_crl_dcc as m  # noqa: WPS433 (import inside fixture)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f'cannot import continual_crl_dcc with stubs: {e!r}')
    return m


class _Args:
    """Minimal stand-in exposing only the fields the ckpt helpers read."""
    dcc_combine_mode = 'add'
    dcc_goal_encoder_mode = 'shared'
    dcc_use_dyn = True
    rep_size = 64
    dcc_phi_task_width = 256
    dcc_phi_task_depth = 4
    seed = 1


def _actor_state(m):
    actor = m.Actor(action_size=4)
    params = actor.init(jax.random.PRNGKey(0),
                        jnp.ones((1, 8)), jnp.ones((1, 64)))
    tx = optax.adam(3e-4)
    return actor, tx, TrainState.create(apply_fn=actor.apply, params=params, tx=tx)


def _groups():
    tx = optax.adam(1e-3)
    p = {'w': jnp.ones((2, 2))}
    return {
        'b_shared': (p, tx.init(p)), 'h_phi': (p, tx.init(p)),
        'h_dyn': (p, tx.init(p)), 'phi_task': (p, tx.init(p)),
        'psi_shared': (p, tx.init(p)),
        'psi_task': (None, None), 'psi_proj': (None, None),
    }


def test_driver_save_load_autoresume_roundtrip(driver, tmp_path):
    m = driver
    args = _Args()
    base = str(tmp_path)

    actor, _, state = _actor_state(m)
    state = state.apply_gradients(
        grads=jax.tree_util.tree_map(jnp.ones_like, state.params))

    # Save through the driver's own wrapper (note the task_id positional arg).
    m.save_ckpt(base, 0, 'cube-1-task1', args, state, _groups())
    ckpt_path = m._ckpt_path(base, 0, args)
    assert ckpt_path.exists()

    # auto_resume sees exactly one completed task.
    assert m.auto_resume(base, 3, args) == 0

    # load_ckpt validates and returns the payload; reconstruct the actor.
    payload = m.load_ckpt(base, 0, args)
    assert payload is not None and payload['task_idx'] == 0
    restored = m.dcc_checkpoint.restore_actor_state(payload, actor, optax.adam(3e-4))
    for a, b in zip(jax.tree_util.tree_leaves(restored.params),
                    jax.tree_util.tree_leaves(state.params)):
        assert np.allclose(a, b)
    assert int(restored.step) == int(state.step)


def test_driver_autoresume_rejects_zero_byte_task0(driver, tmp_path):
    m = driver
    args = _Args()
    base = str(tmp_path)
    path = m._ckpt_path(base, 0, args)
    os.makedirs(path.parent, exist_ok=True)
    path.write_bytes(b'')  # crashed-run artefact
    assert m.auto_resume(base, 3, args) == -1  # task 0 re-runs, not skipped
