import argparse
import os
import numpy as np
from PIL import Image
import cv2
import torch
import clip
from transformers import pipeline, SegformerImageProcessor, SegformerForSemanticSegmentation
from skimage import measure

parser = argparse.ArgumentParser(description='Apply mask to images using CLIP.')
parser.add_argument('-i', '--image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-p', '--prompt', type=str, required=True, help='Text prompt for segmentation.')
parser.add_argument('-o', '--output', type=str, required=True, help='Directory to save the output mask image.')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

image_name = args.image
text_prompt = args.prompt
output_dir = args.output

# Segmentation
model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
processor = SegformerImageProcessor(do_resize=False)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
original_image = Image.open(image_name)
pixel_values = processor(original_image, return_tensors="pt").pixel_values.to(device)

with torch.no_grad():
  outputs = model(pixel_values)
  logits = outputs.logits

predicted_segmentation_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[original_image.size[::-1]])[0]
predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

color_seg = np.zeros((predicted_segmentation_map.shape[0],
                      predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

palette = np.array(ade_palette())
for label, color in enumerate(palette):
    color_seg[predicted_segmentation_map == label, :] = color

color_seg = color_seg[..., ::-1]
grayscale_seg = cv2.cvtColor(color_seg, cv2.COLOR_BGR2GRAY)
labels = measure.label(grayscale_seg, connectivity=grayscale_seg.ndim)

# Retrieve each segment object separately
segment_objects = []
for label in np.unique(labels):
    if label == 0:
        continue  # Skip background label
    segment_mask = np.zeros_like(grayscale_seg, dtype=np.uint8)
    segment_mask[labels == label] = 255
    segment_objects.append(segment_mask)

with torch.no_grad():
  outputs = model(pixel_values)
  logits = outputs.logits

# CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Text prompt preprocessing for CLIP
text_tokens = clip.tokenize([text_prompt]).to(device)

segment_image = []
for idx, mask in enumerate(segment_objects):
    mask_image = Image.fromarray(segment_objects[idx])
    mask_image = mask_image.convert("L")
    mask_image.save(os.path.join(output_dir, f"mask.png"))
    masked_image = Image.open(os.path.join(output_dir, 'mask.png'))
    segment_image.append(preprocess(masked_image))

segment_image_tensor = torch.stack(segment_image).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    image_features = model.encode_image(segment_image_tensor)
    logits_per_image, logits_per_text = model(image=segment_image_tensor, text=text_tokens)
    probs = logits_per_text.softmax(dim=-1).cpu().numpy()

np.set_printoptions(precision=3, suppress=True)
print("==========================================================================")
print('The probability distribution over the segments are:')
print(probs)

best_mask_idx = np.argmax(probs)
best_mask = Image.fromarray(segment_objects[best_mask_idx]).convert("L")
best_mask.save(os.path.join(output_dir, 'best_segment.png'))

np.set_printoptions(precision=3, suppress=True)
print(f'The max. probability = {probs[0, best_mask_idx]}')
print("==========================================================================")
