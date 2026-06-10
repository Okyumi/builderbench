#!/usr/bin/env python3
"""Enumerate DCC ablation configurations for BuilderBench.

Mirrors the API of ``experiment_configs.py`` so the same SLURM launcher
shape works (``--total`` / ``--setting i`` / ``--list``).

DCC ablations covered:

  baseline_persistent   contrastive_rl baseline; critic_mode=persistent (no DCC)
  baseline_reset        contrastive_rl baseline; critic_mode=reset
  dcc_add_shared        DCC, additive combine, shared psi
  dcc_concat_shared     DCC, concat combine, shared psi
  dcc_no_dyn            DCC additive shared, dynamic loss OFF
  dcc_goal_task         DCC additive, psi is task_specific
  dcc_goal_partial      DCC additive, psi is partial_shared
  dcc_goal_decomposed   DCC additive, psi is decomposed
  dcc_goal_projected    DCC additive, psi is projected

Edit ABLATIONS below to grow / shrink the sweep.
"""
import argparse
import sys
from copy import deepcopy


SEEDS = [1, 2, 3]    # adjust per HPC budget; one seed for smoke testing


def _row(name, *, runner, **kw):
    """name -> (runner, dict of CLI flags). All bool flags must be strings
    'true'/'false' so the shell launcher can branch trivially.
    """
    return name, runner, {k: ('true' if v is True else 'false' if v is False else v)
                          for k, v in kw.items()}


ABLATIONS = [
    # Baselines using the existing R/P/C driver (no DCC).
    _row('baseline_persistent',
         runner='continual_crl.py',
         actor_mode='reset', critic_mode='persistent'),
    _row('baseline_reset',
         runner='continual_crl.py',
         actor_mode='reset', critic_mode='reset'),

    # Core DCC (additive + shared goal encoder).
    _row('dcc_add_shared',
         runner='continual_crl_dcc.py',
         use_dcc=True, dcc_combine_mode='add',
         dcc_goal_encoder_mode='shared',
         dcc_use_dyn=True),

    # Combine mode ablation.
    _row('dcc_concat_shared',
         runner='continual_crl_dcc.py',
         use_dcc=True, dcc_combine_mode='concat',
         dcc_goal_encoder_mode='shared',
         dcc_use_dyn=True),

    # Dynamic-loss ablation.
    _row('dcc_no_dyn',
         runner='continual_crl_dcc.py',
         use_dcc=True, dcc_combine_mode='add',
         dcc_goal_encoder_mode='shared',
         dcc_use_dyn=False),

    # Goal-encoder ablations.
    _row('dcc_goal_task',
         runner='continual_crl_dcc.py',
         use_dcc=True, dcc_combine_mode='add',
         dcc_goal_encoder_mode='task_specific',
         dcc_use_dyn=True),
    _row('dcc_goal_partial',
         runner='continual_crl_dcc.py',
         use_dcc=True, dcc_combine_mode='add',
         dcc_goal_encoder_mode='partial_shared',
         dcc_use_dyn=True),
    _row('dcc_goal_decomposed',
         runner='continual_crl_dcc.py',
         use_dcc=True, dcc_combine_mode='add',
         dcc_goal_encoder_mode='decomposed',
         dcc_use_dyn=True),
    _row('dcc_goal_projected',
         runner='continual_crl_dcc.py',
         use_dcc=True, dcc_combine_mode='add',
         dcc_goal_encoder_mode='projected',
         dcc_use_dyn=True),
]


def build_configs():
    configs = []
    for (name, runner, kw) in ABLATIONS:
        for seed in SEEDS:
            cfg = {'name': name, 'runner': runner, 'seed': seed}
            cfg.update(kw)
            configs.append(cfg)
    return configs


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--setting', type=int)
    g.add_argument('--total', action='store_true')
    g.add_argument('--list', action='store_true')
    a = p.parse_args()

    cfgs = build_configs()
    if a.total:
        print(len(cfgs))
        return
    if a.list:
        print(f'Total: {len(cfgs)}')
        for i, c in enumerate(cfgs):
            extras = ' '.join(f'{k}={v}' for k, v in c.items()
                              if k not in ('name', 'runner', 'seed'))
            print(f'{i:3d} {c["runner"]:<22} name={c["name"]:<24} '
                  f'seed={c["seed"]} {extras}')
        return

    idx = a.setting
    if idx < 0 or idx >= len(cfgs):
        print(f'ERROR: setting {idx} out of range [0, {len(cfgs)-1}]',
              file=sys.stderr)
        sys.exit(1)
    for k, v in cfgs[idx].items():
        if isinstance(v, bool):
            v = 'true' if v else 'false'
        print(f'{k.upper()}={v}')


if __name__ == '__main__':
    main()
