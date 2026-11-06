# Contributing

## Setup

You need Python 3.12, an NVIDIA GPU, and access to the gated [facebook/sam3](https://huggingface.co/facebook/sam3) weights (`hf auth login` after you're approved).

```
git clone --recursive https://github.com/jayhemnani9910/recognize && cd recognize
pip install torch torchvision einops decord pycocotools psutil matplotlib -e ./sam3
```

If you cloned without `--recursive`, run `git submodule update --init` first.

## Checks

There are no tests yet. CI runs ruff and checks that the install resolves and the imports load. Run the same lint locally:

```
pip install ruff==0.16.10
ruff check run_sam3_image.py viz_sam3_image.py
ruff format --check run_sam3_image.py viz_sam3_image.py
```

If you change what the scripts do, run them on an image in `data/image.jpeg` and say in the PR what you saw.

## Style

Keep the scripts short and readable top to bottom. Let `ruff format` handle formatting. Don't commit images, weights or anything under `data/`.

## Branches and PRs

Branch off `master`, keep one change per PR, and fill in the PR template. CI has to pass before I merge.

## Bugs

Open an issue with the bug template: the command you ran, what you expected, what happened (full traceback), your GPU, OS, Python and PyTorch versions.

I try to reply to issues within a few days.
