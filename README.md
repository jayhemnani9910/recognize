# recognize

SAM3 (Segment Anything Model) recognition and segmentation experiments.

## Notes

- Main docs live under `models/facebook-sam3/README.md`.

## Setup

```
git submodule update --init
pip install -e sam3 matplotlib
```

`build_sam3_image_model()` downloads the `facebook/sam3` checkpoint from Hugging Face (`load_from_HF=True` by default). The checkpoint is gated: request access on the model page, then `hf auth login`.

Put an input image at `data/image.jpeg` (the `data/` folder is gitignored).

## Run

```
python run_sam3_image.py   # prints mask count and box shape for the prompt "person"
python viz_sam3_image.py   # overlays the masks and saves data/output_masks.png
```
