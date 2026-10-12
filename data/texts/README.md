# data/texts

Class names, their frozen BioMedCLIP embeddings and the class metadata used by
CytoSPM. No images or annotations are included.

| File | Contents |
|---|---|
| `tct_ngc_fullnames_30.json` | Class-name prompts of the 30 training classes (24 base + 6 negative), in category-id order |
| `tct_ngc_fullnames_30_embeddings_biomedclip.pth` | BioMedCLIP embeddings of these class names |
| `tct_ngc_novel_merged_9.json` | Class-name prompts of the 9 novel classes |
| `tct_ngc_novel_merged_9_emb_biomedclip.pth` | BioMedCLIP embeddings of the novel class names |
| `tct_ngc_class_organ_mask_base30.pt` | Class-to-organ mask of the training classes |
| `tct_ngc_class_organ_mask_novel_merged.pt` | Class-to-organ mask of the novel classes |
| `tct_ngc_taxonomy.json` | Organs, classes and reporting-system categories |
| `tct_ngc_dev30_negatives.json` | Negative classes excluded from the base metric |

Embeddings are L2-normalized outputs of the frozen text encoder of
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
(revision `9f341de24bfb00180f1b847274256e9b65a3a32e`). Rebuild them, or encode
the class names of another dataset, with:

```bash
python tools/build_text_embeddings.py class-names data/texts/tct_ngc_fullnames_30.json out.pth
```

Organ masks are built from an annotation file and the taxonomy with
`tools/build_class_organ_mask.py`.
