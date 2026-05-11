# Three Ways to Rewrite a Photograph

CS 5788 (Introduction to Generative Models, Cornell Tech, Spring 2026) final project.

Three text-and-mask-driven editing operations on real photographs, exposed through one Gradio web UI:

- **Object Replacement** — swap one object in a photo for another from text prompts. SD 1.5 + DDIM inversion + null-text optimization + a per-token attention swap schedule + pixel composite over an attention-derived mask.
- **Object Relocation** — move an object within the same frame to a user-painted target location. SD 2.1 + DDPM inversion + per-step source-to-target noise-prior shift + SDEdit harmonization with background and target locks + SD 2.1 inpainting cleanup of the vacated source region.
- **Style Transfer** — repaint the photo in one or more painterly styles. SD 1.5 + per-style LoRA adapters trained on WikiArt (Van Gogh, Seurat, Monet, Ukiyo-e, Picasso) + multi-adapter blending at inference.

Each module loads its own checkpoints. All three weight sets fit comfortably on a single A100. See [Module ownership](#module-ownership) below for who built what.

## Demo

```bash
pip install -r requirements.txt
pip install -r relocate/requirements.txt
pip install -r style/requirements.txt
python platform/app.py
```

The Gradio app prints a `*.gradio.live` URL. Three tabs, cascading hand-offs (Replace → Move → Style). On Colab use the runnable [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb) which clones, installs, and demos all three modules end-to-end on one image.

## Repo layout

```text
.
├── platform/app.py            # unified Gradio app (three tabs + cascading hand-offs)
├── notebooks/walkthrough.ipynb # end-to-end Colab demo across all three modules
├── report/                    # CVPR-format final report (LaTeX source + figures)
├── requirements.txt           # root deps (replace + style + platform)
│
├── replace/                   # object-replacement module
│   ├── sd_components.py       # SD 1.5 component loader
│   ├── ddim.py                # DDIM forward + reverse, written from scratch
│   ├── null_text_inv.py       # null-text inversion + sampling helper
│   ├── attention_store.py     # cross-attention + schedule controllers
│   ├── schedules.py           # the per-token swap schedules (5 shapes, 3 roles)
│   ├── editor.py              # the Editor class -- main entry point
│   ├── masks.py               # attention-derived localization mask
│   ├── inpaint.py             # blended-diffusion inpaint for shape-mismatch edits
│   └── metrics.py             # CLIP directional similarity, LPIPS variants
│
├── relocate/                  # object-relocation module
│   ├── pipeline/relocation_pipeline.py   # imported by platform
│   ├── inversion/ddpm_inversion.py
│   ├── noise_shift/noise_shift.py
│   ├── utils/                 # image + mask helpers
│   ├── eval/                  # metrics + perceptual loss + visualization
│   ├── quick_test.py          # the synthetic move benchmark cited in the report
│   ├── data/results/          # PNGs the report's relocation figure pulls from
│   └── requirements.txt
│
├── style/                     # style-transfer module
│   ├── inference.py           # imported by platform
│   ├── styles.py              # 5-style registry imported by platform
│   ├── train_lora.py          # documents the training recipe in the report
│   ├── train_all.py           # batch wrapper around train_lora.py
│   ├── output/lora/           # trained per-style LoRA weights (5 styles)
│   └── requirements.txt
│
├── scripts/                   # CLI tools that reproduce the replacement-side numbers
│   ├── reconstruct.py
│   ├── edit_single.py
│   ├── run_ablation.py
│   ├── fetch_real_photos.py
│   └── caption_real_photos.py
│
├── data/
│   ├── demo/headline.jpg      # the curated demo photo for the notebook
│   └── real/                  # ~20 COCO photos (gitignored, fetched on demand)
│
└── outputs/                   # gitignored: ablation grids, metrics CSV, null-text cache
```

## Reproducing the report's replacement-side numbers

```bash
# 1. fetch ~20 photos from COCO val2017 + write a starter prompts.json
python scripts/fetch_real_photos.py

# 2. upgrade prompts.json with BLIP captions (real-photo inversion needs scene context)
python scripts/caption_real_photos.py

# 3. run the full schedule x mask-mode ablation; produces metrics.csv + per-image grids
python scripts/run_ablation.py
```

Outputs land in `outputs/ablation/`. The runnable end-to-end demo across all three modules is in [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb).

## Headline results (object replacement)

| Metric | Target (proposal) | Achieved |
| --- | --- | --- |
| Reconstruction LPIPS (generated, in-distribution) | < 0.05 | **0.017** |
| CLIP directional similarity (cat → dog edit) | > 0.20 | **0.55** |
| Background LPIPS (mask + pixel composite, cat→dog) | < 0.10 | **0.019** |
| Background LPIPS (averaged across 20 BLIP-captioned COCO photos) | n/a | **0.008 ± 0.011** |

Full discussion (including the relocation and style results) is in [report/](report/).

## Module ownership

| Module | Author | Code under |
| --- | --- | --- |
| Object Replacement | Stefan Arni Arnarsson | `replace/`, `scripts/` |
| Object Relocation | Aryaan Verma | `relocate/` |
| Style Transfer | Chien-Wei Wang | `style/` |
| Unified Gradio platform | Stefan Arni Arnarsson | `platform/` |
| Final report | All three | `report/` |
