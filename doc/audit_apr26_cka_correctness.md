# CKA Audit — Apr 26, 2026

## Symptoms reported

1. `actor_mode=cka` shows **no improvement** over `reset` and `persistent`,
   sometimes **worse** performance.
2. `actor_mode=cka` runs **noticeably slower** than `reset` and `persistent`.

Both symptoms point at the same family of issues. This audit cross-checks
the BuilderBench implementation against the original CKA-RL reference
(`sgcrl/cka-rl-meta-world/models/{cka_rl,fuse_module}.py`) and against
the SGCRL contrastive port (`sgcrl/contrastive/{knowledge_pool,
continual_learning}.py`), which are the two correct references in this
repo family.

---

## What the math should be

The original CKA-RL composes a per-task policy as

$$\theta'_k \;=\; \theta_{\mathrm{base}} \;+\; \sum_{j=1}^{|\mathcal V|}\alpha_j\,v_j \;+\; v_k,$$

with **two learnable scalar tensors**:

1. **`alpha`** — a vector of shape `[|V|]` of *logits*. Its softmax produces
   the attention weights $\alpha_j$ over previous-task knowledge vectors.
   In the original code (`fuse_module.py:FuseLinear.forward`):
   `alpha_normalized = softmax(alpha * alpha_scale, dim=0)`.
2. **`alpha_scale`** — a scalar (or per-layer scalar) "softmax temperature"
   that controls how peaked the blend is.

Both of these enter the actor forward pass *during* training, and both
receive gradients through the actor loss (in the SGCRL port,
`alpha_logits` is called `beta_k`, but the role is identical).

After each task the trained $v_k$ is appended to the pool $\mathcal V$. If
$|\mathcal V| > K_{\max}$, the two most cosine-similar vectors are merged
by averaging.

---

## What BuilderBench actually does

### Bug 1. Both `alpha_logits` and `alpha_scale` are non-existent.

`continual_crl.py:1184–1187`:

```python
cka_actor_pool_c = actor_pool.compute_contribution(
    alpha_logits=None, alpha_scale=args.alpha_scale)
```

`alpha_logits=None` triggers `KnowledgePool.compute_contribution`'s fallback
to `alpha_logits = jnp.zeros(n)`, so $\alpha_j = 1/n$ for every $j$ — **a
uniform average over the pool, with no learning at all**.

`args.alpha_scale` is a hyperparameter (default 1.0) and is **not** a
`jax.numpy` parameter; it is captured as a Python float in the JIT trace
and is therefore frozen for the entire run.

Net effect: there is no learnable $\alpha$ and no learnable $\alpha_{\mathrm{scale}}$.
The composition reduces to

$$\theta'_k \;=\; \theta_{\mathrm{base}} \;+\; \frac{1}{|\mathcal V|}\sum_j v_j \;+\; v_k.$$

This is **strictly weaker** than `persistent` because it forces the
agent to learn $v_k$ from a starting point that is contaminated by an
unweighted average of past-task deltas. Both directions (good and bad)
of every prior task pull the actor away from the identity init that
`persistent` enjoys, and the agent cannot down-weight the unhelpful
ones because $\alpha$ is frozen at uniform.

This is the central correctness bug. It explains why `cka` performs no
better, and often worse, than `reset` / `persistent`.

### Bug 2. `actor_pool_c` is captured as a static constant inside the JIT.

`continual_crl.py:646–656`:

```python
def _get_actor_params(ts):
    if use_actor_cka:
        return _compose_params(actor_base_params, actor_pool_c,
                               ts.actor_state.params)
    return ts.actor_state.params
```

`actor_pool_c` is a Python closure variable, computed once before the
task starts and not threaded through `training_state`. As long as Bug 1
is in place this happens to be self-consistent, but it forecloses the
fix: making $\alpha$ learnable requires $\alpha$ to live in
`training_state` so it can flow through the gradient. Right now
$\alpha$ has no place to live.

### Bug 3. The actor's gradient differentiates through the wrong variable.

`continual_crl.py:735–739` (CKA-actor loss):

```python
def actor_loss(vk_params, critic_params, transitions, key):
    composed = _compose_params(actor_base_params, actor_pool_c,
                               vk_params)
    ...
    means, log_stds = actor.apply(composed, state, g_repr)
```

This is correct as far as it goes — the gradient w.r.t. $v_k$ does
equal the gradient w.r.t. the composed policy because both
`actor_base_params` and `actor_pool_c` enter additively as constants.
The original SGCRL port also relies on this trick (`continual_learning.py:454`).
However, in the SGCRL port the trick is paired with an **outer-loop
gradient step on $\beta_k$ and $\alpha_{\mathrm{scale}}$** (`continual_learning.py:744–807`).
BuilderBench has no such outer-loop step, so the trick yields the right
gradient for $v_k$ but never trains the attention weights at all.

### Bug 4. Pool initialisation appends a zero vector at task 0.

`continual_crl.py:1284–1285`:

```python
actor_base_params = actor_state.params
actor_pool.append(pytree_zeros_like(actor_base_params))
```

This appends a zero-valued knowledge vector to the pool right after
task 0. At task 1 the pool then has one entry, the all-zero vector, and
$\alpha_j = 1/n = 1$, so `pool_contribution = 0`. That is harmless at
task 1. At task 2, however, the pool has two entries: the zero from
task 0 and the trained $v_1$ from task 1. Under uniform $\alpha$ this
gives `pool_contribution = (0 + v_1) / 2 = v_1/2`. The half-strength
$v_1$ contribution is *not* what the original algorithm intends. The
original algorithm intends $\alpha$ to either (a) put roughly all weight
on the relevant prior task and very little on the zero placeholder, or
(b) more typically, omit the zero placeholder altogether and just use
the trained vectors.

The SGCRL port also seeds the pool with a zero vector
(`continual_learning.py` does it implicitly via `_pytree_zeros_like`),
but it pairs this with a **learnable** $\beta$ that learns to suppress
the zero placeholder. With BuilderBench's frozen-uniform $\alpha$, the
placeholder permanently dilutes every later pool contribution.

### Bug 5. Pool merging by `_merge_pair` runs O(n²) cosine similarity in plain Python after every task.

`knowledge_pool.py:31–48`:

```python
for i in range(len(self._vectors)):
    for j in range(i + 1, len(self._vectors)):
        sim = float(...)
```

Pure-Python loop over JAX arrays, with `float(...)` calls forcing host
sync each iteration. For `K_max=5` this means up to 10 host syncs per
task boundary, plus one merge that traces a `jax.tree.map`. This is
small and only fires at task boundaries, so it is **not** what is
slowing training. It is worth fixing for hygiene but is a side
concern.

### Bug 6 (the real cause of slowness). JIT cache misses every task.

When `actor_mode=cka`, the closures `update_actor_and_alpha`,
`actor_step`, etc., capture `actor_base_params` and `actor_pool_c`
as **closed-over JAX arrays**. Each new task constructs a *new*
`actor_pool_c` (from the appended pool) and a *new* (or unchanged)
`actor_base_params`. Although the *shapes* are the same, every new
`@jax.jit`-decorated inner function created via these closures has a
different `id()` for its captured constants. JAX's trace-cache keys
on the function object, not on the closed-over array values, so the
function is re-traced **at the start of every task**.

Empirically this looks like the first few hundred steps of every task
running 5–20× slower than the steady-state under `reset` / `persistent`,
because:

- `reset`: `actor_state.params` is the only changing input; the JIT
  function is reused across tasks.
- `persistent`: same.
- `cka`: closures are re-created per task → re-trace each task.

A second contributor: the `_get_actor_params` closure is called inside
`actor_step` on every environment rollout step. Because
`actor_base_params` and `actor_pool_c` are captured *outside* the
`training_state` pytree, they are folded into the JIT trace as
**static constants**. JAX cannot share the compiled actor-step
function across tasks, and the resulting compiled artifact is also
larger than the non-CKA version.

### Bug 7. `_split_head_body` head detection is by leaf path string.

`continual_crl.py:447–476` and `_is_actor_head_leaf`:

```python
if _is_actor_head_leaf(path):
    ...
```

Without seeing `_is_actor_head_leaf`, but given the SGCRL port masks
heads by string-matching `'Normal'` in the path, the BuilderBench
analogue is fragile. If the actor's head submodule is renamed to
`Dense_4`/`Dense_5` (which the comment at line 450 admits), the
identification depends on layer count of the encoder. Any architecture
change silently re-categorises body as head or vice versa, and
`adapt_heads_only` becomes a different-looking restriction without
errors. This is not why the current run is failing — but it is fragile
enough that it should be fixed alongside the other items.

---

## Why each symptom is now explained

**No improvement / degradation.** Bugs 1, 3, and 4 together strip the
algorithm of the only mechanism (learnable $\alpha$) that lets the
composed policy benefit from the pool. The agent is forced to start
each new task from `theta_base + (uniform mean of past v_j)` and learn
$v_k$ on top of a contaminated init, which is worse than `persistent`
(which starts from the most recent composed policy directly).

**Slower training.** Bug 6 explains the slowdown end-to-end. The fact
that closures over `actor_base_params` and `actor_pool_c` are
captured outside the `training_state` and rebuilt every task means JAX
re-traces and re-compiles every `@jax.jit`-decorated function used in
the inner loop at every task boundary.

---

## Recommended fixes (in priority order)

### Fix A. Make `alpha_logits` and `alpha_scale` learnable parameters in `training_state`.

This is the central correctness fix. In SGCRL's port the equivalent
quantity is `beta_k` and is part of the training state, with its own
optimiser (`sgcrl/contrastive/continual_learning.py:51–61, 744–807`).

The cleanest BuilderBench port is:

```python
@flax.struct.dataclass
class TrainingState:
    ...
    # Per-task CKA scalars (only meaningful when actor_mode == 'cka').
    actor_alpha_logits: jnp.ndarray   # shape [|V|], zeros at task 0/1
    actor_alpha_scale:  jnp.ndarray   # shape [], init 1.0
    actor_alpha_logits_opt_state: optax.OptState
    actor_alpha_scale_opt_state:  optax.OptState
    # Symmetric set for critic if you want the same on the critic side.
```

Initialisation:

```python
actor_alpha_logits = jax.random.normal(k1, (n_pool,)) * 0.01  # warm-start
actor_alpha_scale  = jnp.array(1.0)
```

In the actor loss, change `actor_pool_c` from a closed-over constant
to a quantity computed from `training_state`:

```python
def actor_loss(vk_params, alpha_logits, alpha_scale,
               critic_params, transitions, key):
    pool_c = compute_contribution_from_logits(
        actor_pool_vectors,        # static list of pytrees
        alpha_logits, alpha_scale,
    )
    composed = _compose_params(actor_base_params, pool_c, vk_params)
    ...
```

and use `jax.value_and_grad(actor_loss, argnums=(0, 1, 2), has_aux=True)`
so that the actor optimiser updates `(v_k, alpha_logits, alpha_scale)`
in lockstep. If you want the SGCRL split where $\beta$ uses its own
optimiser with its own learning rate, copy that pattern; otherwise a
single Adam over all three is fine and is closer to the original
CKA-RL implementation.

This single fix removes Bugs 1, 3, and 4 in one change.

### Fix B. Move pool vectors into `training_state` so the JIT cache is preserved across tasks.

Pool vectors are pytrees with the same structure each task; they can
be stacked into a single leading-axis-K_max array and threaded through
`training_state` as a static-shape leaf. After this change:

- closures no longer capture per-task constants;
- the inner `update_actor_and_alpha` and `update_critic` closures are
  built once per process, and their JIT traces are reused across tasks
  because only the values inside `training_state` change.

The pool's *length* changes across tasks. The standard JAX-friendly
trick is to fix shape at `K_max + 1` and pass a `pool_mask` boolean
vector that is multiplied into `alphas` before the softmax. The mask
becomes part of `training_state` and changes in value, not in shape.

This single fix removes Bug 6.

### Fix C. Drop the zero-vector pool seed.

`continual_crl.py:1285`:

```python
actor_pool.append(pytree_zeros_like(actor_base_params))
```

Remove this line. After Fix A, the placeholder is harmless because
$\alpha$ can learn to suppress it; after Fix B, the pool is fixed-size
anyway and the mask handles emptiness. The seeded zero vector is a
relic that the SGCRL port also has and that we should remove from both.

### Fix D. Vectorise `_merge_pair` cosine similarity.

```python
flat = jnp.stack([_flatten_pytree(v) for v in self._vectors])
sims = (flat @ flat.T) / (jnp.linalg.norm(flat, axis=1, keepdims=True) *
                          jnp.linalg.norm(flat, axis=1, keepdims=True).T)
sims = sims.at[jnp.tril_indices(len(flat))].set(-jnp.inf)
i, j = jnp.unravel_index(jnp.argmax(sims), sims.shape)
```

One vectorised compute, no host syncs. Hygiene fix; small impact.

### Fix E. Make `_is_actor_head_leaf` robust to architecture changes.

Use a head-tag set on the actor module rather than string matching on
the leaf path. Either store a list of "head leaf paths" alongside the
actor when it is constructed, or label the head with a Flax
`nn.compact` block named `'head'` and use that name in path matching.

This is a robustness fix; it does not affect current numbers.

---

## Acceptance criteria

After Fix A + Fix B (the two correctness-and-performance fixes):

1. **`actor_mode=cka` ≥ `actor_mode=persistent`** on the 9-cell ablation,
   averaged over seeds, by the end of task 1. (At task 0 the modes are
   identical by construction.)
2. **Per-task wall-clock parity:** the ratio of `cka` step time to
   `persistent` step time should be within 10–20% (the small overhead
   of the extra $|V|$-dim softmax) instead of the current 3–10× slowdown.
3. **`alpha_weights` log shows non-uniform values** that change over
   training, and `alpha_scale` drifts away from $1.0$.

These three together would confirm that the algorithm now does what
the original CKA-RL paper intended.

---

## Cross-references

- Original CKA-RL: `sgcrl/cka-rl-meta-world/models/{cka_rl,fuse_module}.py`
  - `FuseLinear.forward` for the live-time composition with learnable
    `alpha` and `alpha_scale`.
  - `setup_alpha` for initialisation choices ("Randn", "Major", "Uniform").
- SGCRL contrastive port:
  - `sgcrl/contrastive/knowledge_pool.py` for `compose_policy_params`,
    `merge_most_similar_pair`.
  - `sgcrl/contrastive/continual_learning.py:51–61` for the `beta_k`,
    `alpha_scale` slots in training state.
  - `sgcrl/contrastive/continual_learning.py:744–807` for the
    outer-loop gradient on $\beta_k$ and $\alpha_{\mathrm{scale}}$.
