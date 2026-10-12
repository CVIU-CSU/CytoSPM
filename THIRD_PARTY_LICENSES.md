# Third-Party Licenses and Provenance

CytoSPM is released under GPL-3.0 (see [`LICENSE`](LICENSE)). This inventory is
about code that is copied, modified, or redistributed by the release. It is not a claim that every package installed by
the environment is bundled here.

Complete license texts for the components whose source or files are shipped are
under [`third_party_licenses/`](third_party_licenses/). The immutable source
references below are the snapshots used for the release audit; a snapshot pin is
not a claim that it is the original historical fork point.

| Component | License | Audited source / revision | What is present in CytoSPM | Changes / redistribution |
|---|---|---|---|---|
| WeDetect | GPL-3.0 | [WeChatCV/WeDetect@1903f095](https://github.com/WeChatCV/WeDetect/tree/1903f09542ef39feae1a962da193f4c602d9899f) | Modified detector, dataset and optimizer modules under `cytospm/` | Substantial CytoSPM changes are documented in [`NOTICE`](NOTICE); retained upstream headers are not removed |
| YOLO-World | GPL-3.0 | [YOLO-World@e723fc72](https://github.com/AILab-CVC/YOLO-World/tree/e723fc723c020f2a1def589d4f58345a368eae48) | Upstream design/code lineage incorporated through WeDetect | GPL obligations follow the derivative source; the release does not ship an unmodified YOLO-World checkout |
| MMDetection | Apache-2.0 | [mmdetection@44ebd17b](https://github.com/open-mmlab/mmdetection/tree/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a) | Copied/modified metric and transform portions, plus runtime dependency | Apache notice and license text are retained in `third_party_licenses/OpenMMLab/` |
| MMYOLO | GPL-3.0 | [mmyolo@8c4d9dc5](https://github.com/open-mmlab/mmyolo/tree/8c4d9dc503dc8e327bec8147e8dc97124052f693) | YOLO neck/head/block lineage used by the detector | Distributed only as part of the GPL derivative; no separate MMYOLO package is bundled |
| YOLOv6 | GPL-3.0 | Referenced upstream lineage in `batch_task_aligned_assigner.py` | Algorithmic/source lineage only | The CytoSPM source remains GPL; no YOLOv6 checkout is redistributed |
| ConvNeXt | MIT | [ConvNeXt@048efcea](https://github.com/facebookresearch/ConvNeXt/tree/048efcea897d999aed302f2639b6270aedf8d4c8) | ConvNeXt implementation in `cytospm/models/backbones/mm_backbone.py` | Modified implementation; attribution and MIT terms are in `third_party_licenses/ConvNeXt/` |
| RepVGG | MIT | [RepVGG@eae7c520](https://github.com/DingXiaoH/RepVGG/tree/eae7c5204001eaf195bbe2ee72fb6a37855cce33) | `RepVGGBlock` implementation in `cytospm/models/necks/yolo_world_pafpn.py` | Modified implementation; attribution and MIT terms are in `third_party_licenses/RepVGG/` |
| BiomedCLIP | MIT (model-card declaration) | [HF model@9f341de2](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/tree/9f341de24bfb00180f1b847274256e9b65a3a32e) | External encoder used to produce the class-name embeddings and prompt banks | No BiomedCLIP source or weights are shipped; only derived embedding tensors are present |

## Runtime-only dependencies

MMCV, MMEngine, MMDetection, PyTorch, timm, open_clip and their transitive
dependencies are installed from package distributions according to
`requirements.txt`. They are not vendored into this source archive.
Their package-level license notices remain the responsibility of the corresponding
distribution; the copied OpenMMLab source portions are covered above.

## Prompt and medical references

The morphology prompt text is author-written and distributed as CytoSPM content.
The Bethesda, Paris, respiratory, and serous-fluid books listed in
[`PROMPT_PROVENANCE.md`](PROMPT_PROVENANCE.md) are medical references only. No
book text, figure, or table is redistributed, and no book license is being claimed
for this repository.

## Release obligations

- Preserve this file, [`NOTICE`](NOTICE), the root GPL-3.0 text, and all component
  license texts when redistributing source or binaries.
- Keep the upstream copyright headers and the `Modified by the CytoSPM Authors`
  notices in source files that contain copied code.
- Do not infer that restricted PentaCyto images, annotations, or clinical records
  are covered by GPL-3.0; they are not included in this repository.
