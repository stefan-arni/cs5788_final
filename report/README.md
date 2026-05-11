# Report (CS 5788 final)

CVPR-format 4-page report. Compiled with the official CVPR 2026 LaTeX template.

## Compile (Overleaf)

1. Create a new Overleaf project from the [official CVPR 2026 template](https://github.com/cvpr-org/author-kit).
2. Replace the template's `main.tex`, `preamble.tex`, `sec/`, and `main.bib` with the files in this directory.
3. Keep the template's `cvpr.sty`, `ieeenat_fullname.bst`, and any other support files.
4. Compile with `pdflatex` (the template's default).

## Compile (local)

```bash
# from inside this directory, after copying cvpr.sty + ieeenat_fullname.bst here
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Layout

| File | Owner | Status |
|---|---|---|
| `main.tex` | mixed | single-file report; everything inlined. Replacement-side written; relocation + style subsections marked `\TODO{}` for teammates. Update author placeholders. |
| `preamble.tex` | Stefan | done (defines `\TODO{}` macro) |
| `main.bib` | Stefan | replacement-side refs included; teammates add their own |
| `figures/` | empty | drop the qualitative grids + pipeline figure here |

## Figures still to add

- `figures/teaser.pdf` — pipeline diagram (referenced as `\label{fig:teaser}` in §2)
- `figures/qualitative.pdf` — 3-row qualitative grid for §3.5

Both have `\fbox{...}` placeholders in the LaTeX so the document compiles without them.

## Page-budget check (4-page strict limit excluding refs)

Roughly:
- Abstract + Introduction: ~1 page
- Method: ~1.5 pages (with one figure)
- Experiments: ~1.25 pages (with one figure + two tables)
- Conclusion: ~0.25 page

Should land at exactly 4. If it overflows after teammates fill in §2.5/2.6 and §3.7/3.8, trim Stefan's experiments first (drop bg-LPIPS table, keep schedule ablation table).
