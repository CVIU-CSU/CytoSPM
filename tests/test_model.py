"""CPU smoke tests for the CytoSPM model (run from the repository root)."""
import json

import torch
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.registry import init_default_scope

import cytospm  # noqa: F401
from cytospm.models.dense_heads import StructuralPromptMatching

CONFIG = 'configs/cytospm.py'


def _class_names(path):
    with open(path, encoding='utf-8') as f:
        return [entry[0] for entry in json.load(f)]


def _build(text_embed_path=None, prompt_bank_path=None):
    cfg = Config.fromfile(CONFIG)
    init_default_scope('mmdet')
    if text_embed_path:
        cfg.model.backbone.text_model.text_embed_path = text_embed_path
    if prompt_bank_path:
        cfg.model.bbox_head.head_module.prompt_matching.prompt_bank_path = prompt_bank_path
    return MODELS.build(cfg.model).eval()


def test_structural_prompt_matching_shapes():
    x = torch.randn(2, 512, 8, 8)
    bank = torch.randn(5, 6, 512)
    s3, s2 = StructuralPromptMatching(top_p=0.01, top_q=0.95)(x, bank)
    assert s3.shape == s2.shape == (2, 5, 8, 8)
    s3, s2 = StructuralPromptMatching(use_prototype=False)(x, bank)
    assert s3.shape == (2, 5, 8, 8) and s2 is None


def test_masked_prompts_are_ignored():
    x = torch.randn(1, 512, 4, 4)
    bank = torch.randn(3, 6, 512)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    mask[:, :2] = True
    noisy = bank.clone()
    noisy[:, 2:] = 100 * torch.randn(3, 4, 512)
    a = StructuralPromptMatching(top_q=0, bank_mask=mask)(x, bank)
    b = StructuralPromptMatching(top_q=0, bank_mask=mask)(x, noisy)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


@torch.no_grad()
def test_forward_with_base_and_novel_vocabularies():
    inputs = torch.rand(1, 3, 320, 320)

    base = _build()
    texts = _class_names('data/texts/tct_ngc_fullnames_30.json')
    cls_scores, _ = base._forward(inputs, {'texts': [texts]})
    assert [s.shape[1] for s in cls_scores] == [30, 30, 30]

    novel = _build(
        text_embed_path='data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth',
        prompt_bank_path='data/promptbank/'
        'tct_ngc_descriptors_novel9_structmatch.pt')
    novel.load_state_dict(base.state_dict())
    texts = _class_names('data/texts/tct_ngc_novel_merged_9.json')
    cls_scores, _ = novel._forward(inputs, {'texts': [texts]})
    assert [s.shape[1] for s in cls_scores] == [9, 9, 9]
