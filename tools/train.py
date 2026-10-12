#!/usr/bin/env python
# Copyright (c) The CytoSPM Authors. All rights reserved.
"""Train CytoSPM with the MMEngine runner."""
import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from mmengine.config import Config, DictAction  # noqa: E402
from mmengine.runner import Runner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Train CytoSPM')
    parser.add_argument('config', help='config file')
    parser.add_argument('--work-dir', help='directory for logs and checkpoints')
    parser.add_argument(
        '--resume', nargs='?', type=str, const='auto',
        help='resume from the given checkpoint, or from the latest one in '
        'the work dir if no path is given')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config entries, e.g. '
        'model.bbox_head.head_module.prompt_matching.top_q=0.5')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume
    if args.seed is not None:
        cfg.randomness = dict(cfg.get('randomness', {}), seed=args.seed)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
