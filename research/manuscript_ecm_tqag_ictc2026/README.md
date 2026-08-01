# ECM-TQAG ICTC 2026 manuscript source

This archive contains the LaTeX source of the manuscript and its required bibliography and figure asset.

## Contents

- `main.tex` — IEEE conference manuscript source
- `references.bib` — BibTeX bibliography
- `ecm_tqag_architecture_v2.pdf` — architecture figure used by `main.tex`
- `results_summary.md` — non-sensitive summary and rejection taxonomy for the 72-cell exploratory matrix

## Build

Use a LaTeX distribution that provides `IEEEtran`, `latexmk`, BibTeX, and the packages declared in `main.tex`.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The command produces `main.pdf`. To remove generated build artifacts:

```bash
latexmk -C
```

## Source scope

This package contains only files required to compile the manuscript. The associated research-software repository, code, schemas, and synthetic fixtures are available separately at:

<https://github.com/mxuanvan02/ECM-TQAG>
