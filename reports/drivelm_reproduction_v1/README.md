# DriveLM-nuScenes six-camera reproduction report

This directory contains the illustrated Chinese technical report for the locally executed DriveLM-nuScenes v1.1 reproduction.

- [Final PDF](drivelm_reproduction_technical_report.pdf)
- [LaTeX source](drivelm_reproduction_technical_report.tex)
- [Figure/data generator](generate_report_assets.py)
- [Auditable asset manifest](assets/asset_manifest.json)

The reported 3,355-sample results are from a deterministic, scene-isolated local development split, not the official hidden challenge evaluator. The official validation set has no public answers; its 15,480 generated responses are audited for schema and coverage only.

## Rebuild

Run from the DriveLM repository root:

```bash
python reports/drivelm_reproduction_v1/generate_report_assets.py
cd reports/drivelm_reproduction_v1
xelatex -interaction=nonstopmode -halt-on-error drivelm_reproduction_technical_report.tex
xelatex -interaction=nonstopmode -halt-on-error drivelm_reproduction_technical_report.tex
```

The figure generator reads the real dataset report, metrics, error analysis, training log, and demo videos from the paths recorded in its source. It does not fabricate chart values. The two XeLaTeX passes resolve the table of contents and internal links.

Main development-set results: Exact Match 44.26%, Token-F1 73.33%, ROUGE-L 71.59%, and multiple-choice accuracy 81.46%.
