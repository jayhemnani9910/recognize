from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# build model
model = build_sam3_image_model()
model.to(device)

# build processor
processor = Sam3Processor(model)

# load image
img = Image.open("data/image.jpeg").convert("RGB")

# set image state
state = processor.set_image(img)

# text prompt
output = processor.set_text_prompt(
    state=state,
    prompt="person",
)

masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

print("masks:", len(masks))
print("boxes:", boxes.shape)

