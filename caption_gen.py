import requests
import gpt
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

model_path = "./model"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)


def image_description(raw_image):

    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    return gpt.get_captions(processor.decode(out[0], skip_special_tokens=True))


# # loading image
# raw_image = Image.open("./image1.png").convert('RGB')
# print(image_description(raw_image))
