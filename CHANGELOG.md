# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `run_sam3_image.py`: runs SAM 3 on `data/image.jpeg` with the prompt "person" and prints the mask count and box shape.
- `viz_sam3_image.py`: overlays the masks on the image and saves `data/output_masks.png`.
- `sam3` as a git submodule of facebookresearch/sam3.
- Config, tokenizer files and license from the facebook/sam3 Hugging Face repo under `models/facebook-sam3/`.
- CI: ruff lint and format checks, plus an install and import check.
- CONTRIBUTING.md, issue templates and a PR template.

### Changed
- README rewritten with a tested install line, the GPU and gated-weights requirements, and how the scripts work.
- `.gitignore` now ignores every `.env*` file except `.env.example`.

### Fixed
- Imports sorted and the two scripts formatted with ruff.

### Removed
- The GitHub Pages workflow, which never succeeded.
