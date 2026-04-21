#!/usr/bin/env python3
"""Enumerate experiment configurations for the BuilderBench continual CRL grid.

The 9-configuration ablation grid (reference: sgcrl/experiment_configs.py):
  actor_mode  in {reset, persistent, cka}
  critic_mode in {reset, persistent, cka}

Each configuration is repeated across multiple seeds.

USAGE
-----
  # Total number of configurations (used by the SLURM launcher):
  python experiment_configs.py --total

  # Print a single config's shell-sourceable key=value pairs:
  python experiment_configs.py --setting 0

  # Print all configs as a human-readable table:
  python experiment_configs.py --list

GRID CUSTOMISATION
------------------
Edit ACTOR_MODES / CRITIC_MODES / SEEDS below to change the sweep.

Per-config overrides that are NOT in the base SLURM defaults can be placed
in EXTRA_OVERRIDES (merged into every config) or by giving a config a
`overrides` key of the same name (see build_configs() for an example).

The negative-bank knobs (neg_bank_mode etc.) are included here as
placeholders so the same config file works once the negative bank is
plumbed into `rl/impls/continual_crl.py`. They are silently ignored by
`draft_4.sh` when the upstream driver has not yet implemented them.
"""
import argparse
import itertools
import sys


# =====================================================================
# Grid definition -- edit these to change the experiment sweep
# =====================================================================

ACTOR_MODES = ['reset', 'persistent', 'cka']
CRITIC_MODES = ['reset', 'persistent', 'cka']
SEEDS = [1]

# Per-run overrides that are applied to every configuration in the grid.
# Keys match the lowercase flag names (without leading `--`) exposed by
# `continual_crl.py` (tyro Args dataclass). Values are passed verbatim.
#
# For the BuilderBench 9-cell grid you normally do NOT need any overrides
# here -- the SLURM launcher already sets sensible defaults. Use this for
# quick debug sweeps or targeted overrides.
#
# Examples:
#   EXTRA_OVERRIDES = {'steps_per_task': 5_000_000}   # shorter runs
#   EXTRA_OVERRIDES = {'adapt_heads_only': False}     # full-policy CKA
#
EXTRA_OVERRIDES = {}


# =====================================================================
# Optional: negative-bank placeholders
# ---------------------------------------------------------------------
# The BuilderBench `continual_crl.py` driver does NOT currently expose
# negative-bank flags. Setting NEG_BANK_ENABLED = True here switches the
# placeholder values below on. When the driver is updated to accept
# `--neg_bank_*` flags (mirroring sgcrl), the launcher will pick them up
# automatically. Until then these values are ignored.
# =====================================================================

NEG_BANK_ENABLED = False

NEG_BANK_OVERRIDES = {
    'neg_bank_mode': 'hard_weighted',   # {off, vanilla, hard_weighted}
    'neg_bank_per_task_capacity': 10_000,
    'neg_bank_n_per_step': 256,
    'neg_bank_candidate_pool': 1024,
    'neg_bank_weight': 0.3,
    'neg_bank_max_tasks': 20,
}


# =====================================================================
# Grid enumeration
# =====================================================================

def build_configs():
    """Return the ordered list of configurations (one dict per run)."""
    configs = []
    for actor_mode, critic_mode in itertools.product(
        ACTOR_MODES, CRITIC_MODES
    ):
        for seed in SEEDS:
            cfg = {
                'actor_mode': actor_mode,
                'critic_mode': critic_mode,
                'seed': seed,
            }
            cfg.update(EXTRA_OVERRIDES)
            if NEG_BANK_ENABLED:
                cfg.update(NEG_BANK_OVERRIDES)
            configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Enumerate experiment configurations.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--setting', type=int,
        help='Print the config for this index (0-based) as KEY=VALUE lines.'
    )
    group.add_argument(
        '--total', action='store_true',
        help='Print total number of configs.'
    )
    group.add_argument(
        '--list', action='store_true',
        help='Print all configs as a table.'
    )
    args = parser.parse_args()

    configs = build_configs()

    if args.total:
        print(len(configs))
        return

    if args.list:
        print(f'Total: {len(configs)} configurations')
        header = (
            f'{"idx":>4}  {"actor_mode":<12} {"critic_mode":<12} '
            f'{"seed":>4}'
        )
        print(header)
        print('-' * len(header))
        for i, c in enumerate(configs):
            print(
                f'{i:4d}  {c["actor_mode"]:<12} {c["critic_mode"]:<12} '
                f'{c["seed"]:4d}'
            )
        return

    # --setting N: emit KEY=VALUE pairs (shell-sourceable with eval).
    idx = args.setting
    if idx < 0 or idx >= len(configs):
        print(
            f'ERROR: setting {idx} out of range [0, {len(configs) - 1}]',
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = configs[idx]
    for key, value in cfg.items():
        if isinstance(value, bool):
            # Emit booleans as true/false so the shell can branch on them.
            value = 'true' if value else 'false'
        print(f'{key.upper()}={value}')


if __name__ == '__main__':
    main()
