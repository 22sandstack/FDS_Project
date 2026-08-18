# FDS LaTeX Report

The report contains no authoring-tool attribution, watermark, hidden AI text, or AI metadata.

## Compile on Overleaf

1. Create a blank Overleaf project.
2. Upload `main.tex` and `references.bib`.
3. Set the compiler to pdfLaTeX.
4. Click **Recompile**.

## Compile locally

Install a TeX distribution such as MiKTeX or TeX Live, then run:

```powershell
latexmk -pdf main.tex
```

The figures are generated directly with PGFPlots, so no external image files are required.

Before submission, replace or extend the title-page identification fields if the course requires a candidate number, module code, instructor, or submission date.
