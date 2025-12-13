from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import torch, matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_sam3_image_model().to(device)
processor = Sam3Processor(model)

img = Image.open("data/image.jpeg").convert("RGB")

# MUST be named argument
state = processor.set_image(image=img)

out = processor.set_text_prompt(
    state=state,
    prompt="person"
)

masks = out["masks"]

fig, ax = plt.subplots()
ax.imshow(img)

for m in masks:
    mask = m.squeeze().cpu().numpy()  # (H, W)
    ax.imshow(mask, alpha=0.4)

ax.axis("off")
plt.tight_layout()
plt.savefig("data/output_masks.png")
print("saved → data/output_masks.png")

