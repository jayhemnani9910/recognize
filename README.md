# recognize

[![CI](https://github.com/jayhemnani9910/recognize/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jayhemnani9910/recognize/actions/workflows/ci.yml)

Two short Python scripts that run Meta's SAM 3 on an image and mask every match for a text prompt.

```
git clone --recursive https://github.com/jayhemnani9910/recognize && cd recognize && pip install torch torchvision einops decord pycocotools psutil matplotlib -e ./sam3
```

You need an NVIDIA GPU. SAM 3's model builder puts tensors on CUDA, so a CPU-only PyTorch fails with "Torch not compiled with CUDA enabled".

The weights are gated. Request access on the [facebook/sam3](https://huggingface.co/facebook/sam3) model page, then run `hf auth login`. The first run downloads `sam3.pt` (3.45 GB). Without access the scripts stop with a `GatedRepoError`.

## Usage

Put an image at `data/image.jpeg` (`data/` is gitignored), then:

```
python run_sam3_image.py   # prints the mask count and box shape for the prompt "person"
python viz_sam3_image.py   # overlays the masks on the image and saves data/output_masks.png
```

The prompt is the `prompt="person"` line in each script. Change it to any short phrase, like "dog" or "red car".

## How it works

Both scripts do the same four things:

1. `build_sam3_image_model()` builds SAM 3 and pulls the checkpoint from Hugging Face.
2. `Sam3Processor.set_image()` encodes the image once.
3. `set_text_prompt()` finds every instance of the phrase and returns `masks`, `boxes` and `scores`. Detections under the processor's 0.5 confidence threshold are dropped.
4. `run_sam3_image.py` prints the result. `viz_sam3_image.py` draws each mask at 40% opacity with matplotlib.

`sam3/` is a git submodule of [facebookresearch/sam3](https://github.com/facebookresearch/sam3), pinned at `84cc43b`. Its own `pyproject.toml` leaves out torch, torchvision, einops, decord, pycocotools and psutil, which is why the install line lists them.

`models/facebook-sam3/` holds the config, tokenizer files, model card and license from the Hugging Face model repo. The scripts don't read these files. They load everything through the `sam3` package.

## Credits and license

The two scripts are MIT, see [LICENSE](LICENSE).

SAM 3 is by Meta AI Research. Its code (the `sam3/` submodule), the `sam3.pt` weights and the files in `models/facebook-sam3/` are under Meta's [SAM License](models/facebook-sam3/LICENSE), which has its own terms and use restrictions. MIT does not cover them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
