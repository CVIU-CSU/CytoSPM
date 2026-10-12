#!/usr/bin/env python
# Copyright (c) The CytoSPM Authors. All rights reserved.
"""Zero-shot evaluation on novel categories.

The detector weights stay fixed; only the class-name embeddings, the prompt
bank and the organ mask are replaced by those of the novel vocabulary. The
defaults point to the PentaCyto novel split.
"""
import argparse
import json
import os.path as osp
import sys
from pathlib import Path

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402

from cytospm.utils import (derive_metainfo, resolve_ann_file,  # noqa: E402
                           resolve_eval_checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CytoSPM (novel split)')
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data-root', help='dataset root (overrides the config)')
    parser.add_argument(
        '--ann-file', default='annotations/instances_test_novel_merged_9.json',
        help='annotation file, relative to the dataset root')
    parser.add_argument(
        '--class-names', default='data/texts/tct_ngc_novel_merged_9.json',
        help='class-name prompts, one per category in category-id order')
    parser.add_argument(
        '--class-embeddings',
        default='data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth',
        help='BioMedCLIP embeddings of the class names')
    parser.add_argument(
        '--prompt-bank',
        default='data/promptbank/tct_ngc_descriptors_novel9_structmatch.pt',
        help='encoded structured prompt bank of the novel classes')
    parser.add_argument(
        '--organ-mask', default='data/texts/tct_ngc_class_organ_mask_novel_merged.pt',
        help='class-to-organ mask of the novel classes')
    parser.add_argument('--no-organ-mask', action='store_true',
                        help='do not restrict scores to the image organ')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--work-dir', default='./work_dirs/test_novel')
    parser.add_argument('--metrics-out', help='write the metrics to this JSON file')
    parser.add_argument('--outfile-prefix', help='save predictions to <prefix>.bbox.json')
    parser.add_argument(
        '--allow-unsafe-checkpoint', action='store_true',
        help='load a full training checkpoint you created yourself')
    parser.add_argument('--launcher', default='none', choices=['none', 'pytorch'])
    return parser.parse_args()


def load_organ_mask(path, classes):
    """Load an organ mask and order its rows like ``classes``."""
    pkg = torch.load(path, weights_only=True, map_location='cpu')
    mask = pkg['mask']
    names = list(pkg.get('class_names', classes))
    if names != list(classes):
        index = {name: i for i, name in enumerate(names)}
        missing = [c for c in classes if c not in index]
        if missing:
            raise SystemExit(f'organ mask {path} lacks classes {missing}')
        mask = mask[[index[c] for c in classes]]
    return mask


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = resolve_eval_checkpoint(
        args.checkpoint, allow_unsafe_checkpoint=args.allow_unsafe_checkpoint)

    data_root = args.data_root or cfg.data_root
    ann_path = resolve_ann_file(data_root, args.ann_file)
    classes, num_classes = derive_metainfo(ann_path)
    with open(args.class_names, encoding='utf-8') as f:
        num_prompts = len(json.load(f))
    bank = torch.load(args.prompt_bank, map_location='cpu', weights_only=True)
    if not num_prompts == bank['emb'].shape[0] == num_classes:
        raise SystemExit(
            f'{num_classes} categories in {ann_path}, but {num_prompts} class '
            f'names and {bank["emb"].shape[0]} prompt-bank classes')

    cfg.model.backbone.text_model = dict(
        type='PseudoLanguageBackbone', text_embed_path=args.class_embeddings)
    cfg.model.bbox_head.head_module.prompt_matching.prompt_bank_path = args.prompt_bank

    cfg.test_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='MultiModalDataset',
            dataset=dict(
                type='WeCocoDataset',
                data_root=data_root,
                test_mode=True,
                ann_file=args.ann_file,
                data_prefix=dict(img='images/'),
                batch_shapes_cfg=None,
                metainfo=dict(classes=classes)),
            class_text_path=args.class_names,
            pipeline=cfg.test_pipeline))
    cfg.test_evaluator = dict(
        type='CocoMetric', ann_file=str(ann_path), metric='bbox', classwise=True)
    if args.outfile_prefix:
        cfg.test_evaluator['outfile_prefix'] = args.outfile_prefix
    cfg.work_dir = args.work_dir

    runner = Runner.from_cfg(cfg)
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    if args.no_organ_mask:
        model.bbox_head.set_organ_class_mask(None)
    else:
        model.bbox_head.set_organ_class_mask(load_organ_mask(args.organ_mask, classes))

    metrics = runner.test()
    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, default=float))
        print(f'metrics written to {out}')


if __name__ == '__main__':
    main()
