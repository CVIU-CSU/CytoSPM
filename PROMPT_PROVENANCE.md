# Prompt Bank Provenance

## Authorship and scope

The structured cytomorphology prompts in `data/promptbank/` were written by the CytoSPM authors for this project. They are
original summaries of diagnostic morphology and dataset-specific class merges,
informed by the reporting systems listed below. They are not copied passages,
translations, OCR output, or reproductions of any book's figures or tables.

The cited books describe medical facts, terminology and diagnostic criteria. They
are bibliography, not software components redistributed by this repository. Their
mention does not place the books under the repository's GPL-3.0 license and does not
require their texts to be included in `third_party_licenses/`. The authored prompt
text itself is distributed as part of CytoSPM under the repository license.

The `.pt` prompt banks contain BioMedCLIP encodings of these prompts and can be
rebuilt with `tools/build_text_embeddings.py`.

## Reference systems

1. Nayar R, Wilbur DC, editors. *The Bethesda System for Reporting Cervical
   Cytology: Definitions, Criteria, and Explanatory Notes*. 3rd ed. Springer;
   2015. DOI: `10.1007/978-3-319-11074-5`. ISBN: `978-3-319-11073-8`
   (print), `978-3-319-11074-5` (electronic).
2. Ali SZ, VanderLaan PA, editors. *The Bethesda System for Reporting Thyroid
   Cytopathology: Definitions, Criteria, and Explanatory Notes*. 3rd ed.
   Springer; 2023. DOI: `10.1007/978-3-031-28046-7`. ISBN:
   `978-3-031-28045-0` (print), `978-3-031-28046-7` (electronic).
3. Layfield LJ, Baloch Z, editors. *The Papanicolaou Society of Cytopathology
   System for Reporting Respiratory Cytology*. Springer; 2019. DOI:
   `10.1007/978-3-319-97235-0`. ISBN: `978-3-319-97234-3` (print),
   `978-3-319-97235-0` (electronic).
4. Chandra A, Crothers B, Kurtycz D, Schmitt F, editors. *The International
   System for Serous Fluid Cytopathology*. Springer; 2020. DOI:
   `10.1007/978-3-030-53908-5`. ISBN: `978-3-030-53907-8` (print),
   `978-3-030-53908-5` (electronic).
5. Wojcik EM, Kurtycz DFI, Rosenthal DL, editors. *The Paris System for
   Reporting Urinary Cytology*. 2nd ed. Springer; 2022. DOI:
   `10.1007/978-3-030-88686-8`. ISBN: `978-3-030-88685-1` (print),
   `978-3-030-88686-8` (electronic).
