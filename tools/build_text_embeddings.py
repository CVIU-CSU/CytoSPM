#!/usr/bin/env python
# Copyright (c) The CytoSPM Authors. All rights reserved.
"""Encode class names or a structured prompt bank with frozen BioMedCLIP.

Class names (input: JSON list of ``[name]`` entries, one per class)::

    python tools/build_text_embeddings.py class-names \\
        data/texts/tct_ngc_fullnames_30.json my_class_names.pth

Prompt bank (input: JSON whose ``prompts`` maps each class to a list of short
morphological phrases; the number of phrases may differ between classes)::

    python tools/build_text_embeddings.py prompt-bank \\
        data/promptbank/tct_ngc_prompts_base30.json my_prompt_bank.pt

Embeddings are L2-normalized. The class-name output maps each name to a
``[512]`` vector. The prompt-bank output stores ``emb`` ``[C, M, 512]`` (zero
padded to the longest class), ``mask`` ``[C, M]``, ``labels`` and
``phrase_counts``, in the order of the ``prompts`` mapping.
"""
import argparse
import json

import torch
import torch.nn.functional as F

BIOMEDCLIP = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'


def load_encoder():
    import open_clip
    model, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP)
    return model.eval(), open_clip.get_tokenizer(BIOMEDCLIP)


@torch.no_grad()
def encode(model, tokenizer, texts, batch_size=256):
    out = []
    for i in range(0, len(texts), batch_size):
        emb = model.encode_text(tokenizer(texts[i:i + batch_size])).float()
        out.append(F.normalize(emb, dim=-1))
    return torch.cat(out)


def build_class_names(src, dst):
    names = [entry[0] for entry in json.load(open(src, encoding='utf-8'))]
    model, tokenizer = load_encoder()
    emb = encode(model, tokenizer, names)
    torch.save({name: emb[i] for i, name in enumerate(names)}, dst)
    print(f'saved {len(names)} class-name embeddings to {dst}')


def build_prompt_bank(src, dst):
    prompts = json.load(open(src, encoding='utf-8'))['prompts']
    labels = list(prompts)
    num_slots = max(len(phrases) for phrases in prompts.values())
    flat, pos = [], []
    for ci, label in enumerate(labels):
        for j, phrase in enumerate(prompts[label]):
            pos.append((ci, j))
            flat.append(phrase)
    model, tokenizer = load_encoder()
    embs = encode(model, tokenizer, flat)
    emb = torch.zeros(len(labels), num_slots, embs.shape[-1])
    mask = torch.zeros(len(labels), num_slots, dtype=torch.bool)
    for k, (ci, j) in enumerate(pos):
        emb[ci, j] = embs[k]
        mask[ci, j] = True
    torch.save({'emb': emb, 'mask': mask, 'labels': labels,
                'phrase_counts': mask.sum(1).tolist()}, dst)
    print(f'saved prompt bank {tuple(emb.shape)} ({len(flat)} phrases) to {dst}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('kind', choices=['class-names', 'prompt-bank'])
    parser.add_argument('src', help='input JSON')
    parser.add_argument('dst', help='output .pth / .pt')
    args = parser.parse_args()
    if args.kind == 'class-names':
        build_class_names(args.src, args.dst)
    else:
        build_prompt_bank(args.src, args.dst)


if __name__ == '__main__':
    main()
