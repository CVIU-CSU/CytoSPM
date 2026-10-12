#!/usr/bin/env python
# Copyright (c) The CytoSPM Authors. All rights reserved.
"""Evaluate CytoSPM on the base split.

Reports COCO AP over the 24 base categories with the negative classes
excluded (``coco/all_class/mAP``), plus per-organ and per-class AP.
"""
import argparse
import json
import os.path as osp
import sys
from pathlib import Path

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from mmengine.config import Config, DictAction  # noqa: E402
from mmengine.runner import Runner  # noqa: E402

from cytospm.utils import (load_dev30_negative_classes,  # noqa: E402
                           resolve_ann_file, resolve_eval_checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CytoSPM (base split)')
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data-root', help='dataset root (overrides the config)')
    parser.add_argument(
        '--ann-file', default='annotations/instances_test_base_clean_dev30.json',
        help='annotation file, relative to the dataset root')
    parser.add_argument('--work-dir', default='./work_dirs/test_base')
    parser.add_argument('--metrics-out', help='write the metrics to this JSON file')
    parser.add_argument('--outfile-prefix', help='save predictions to <prefix>.bbox.json')
    parser.add_argument(
        '--allow-unsafe-checkpoint', action='store_true',
        help='load a full training checkpoint you created yourself '
        '(release weights load tensor-only without this flag)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', default='none', choices=['none', 'pytorch'])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = resolve_eval_checkpoint(
        args.checkpoint, allow_unsafe_checkpoint=args.allow_unsafe_checkpoint)

    data_root = args.data_root or cfg.data_root
    for loader in (cfg.val_dataloader, cfg.test_dataloader):
        loader.dataset.dataset.data_root = data_root
        loader.dataset.dataset.ann_file = args.ann_file

    evaluator = cfg.val_evaluator
    evaluator['ann_file'] = str(resolve_ann_file(data_root, args.ann_file))
    evaluator['classwise'] = True
    evaluator['exclude_class_names'] = load_dev30_negative_classes()
    if args.outfile_prefix:
        evaluator['outfile_prefix'] = args.outfile_prefix
    cfg.val_evaluator = evaluator
    cfg.test_evaluator = evaluator
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = args.work_dir

    metrics = Runner.from_cfg(cfg).test()
    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, default=float))
        print(f'metrics written to {out}')


if __name__ == '__main__':
    main()
