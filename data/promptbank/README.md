# data/promptbank

The structured cytomorphology prompt bank used by CytoSPM.

| File | Contents |
|---|---|
| `tct_ngc_prompts_base30.json` | Prompts of the 30 training classes |
| `tct_ngc_prompts_novel9.json` | Prompts of the 9 novel classes |
| `tct_ngc_descriptors_30_structmatch.pt` | BioMedCLIP encoding of `tct_ngc_prompts_base30.json` |
| `tct_ngc_descriptors_novel9_structmatch.pt` | BioMedCLIP encoding of `tct_ngc_prompts_novel9.json` |

Each class has a list of short morphological phrases (cell size and shape,
cytoplasm, nucleus, chromatin, nuclear-to-cytoplasmic ratio, arrangement and
background). The number of phrases differs between classes: up to 22 for the
training classes and up to 23 for the novel classes.

The encoded files store `emb` (`[C, M, 512]`, zero padded to the longest
class), `mask` (`[C, M]`), `labels` (class names) and `phrase_counts`. Rebuild
them, or encode a prompt bank for other classes, with:

```bash
python tools/build_text_embeddings.py prompt-bank data/promptbank/tct_ngc_prompts_base30.json out.pt
```

Re-encoding reproduces the released files up to float32 rounding (maximum
absolute difference below 1e-6).
