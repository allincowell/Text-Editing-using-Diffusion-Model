import argparse
import os
import numpy as np
from PIL import Image
import cv2
import torch
import clip
from transformers import pipeline

def apply_mask(mask_image, output_dir):
    mask_image.save(os.path.join(output_dir, 'mask.png'))
    cv2_mask_image = cv2.imread(os.path.join(output_dir, 'mask.png'), cv2.IMREAD_GRAYSCALE)
    cv2_original_image = cv2.imread(image_name)
    
    masked_image = cv2.bitwise_and(cv2_original_image, cv2_original_image, mask=cv2_mask_image)
    cv2.imwrite(os.path.join(output_dir, 'mask.png'), masked_image)

parser = argparse.ArgumentParser(description='Apply mask to images using CLIP.')
parser.add_argument('-i', '--image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-p', '--prompt', type=str, required=True, help='Text prompt for segmentation.')
parser.add_argument('-o', '--output', type=str, required=True, help='Directory to save the output mask image.')
args = parser.parse_args()

image_name = args.image
text_prompt = args.prompt
output_dir = args.output

# Segmentation
semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
original_image = Image.open(image_name)
segment_masks = semantic_segmentation(original_image)

# CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Text prompt preprocessing for CLIP
text_tokens = clip.tokenize([text_prompt]).to(device)

segment_image = []
for idx, mask in enumerate(segment_masks):
    mask_image = mask['mask']
    apply_mask(mask_image, output_dir)
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
best_mask = segment_masks[best_mask_idx]['mask']
best_mask.save(os.path.join(output_dir, 'best_segment.png'))

np.set_printoptions(precision=3, suppress=True)
print(f'The max. probability = {probs[0, best_mask_idx]}')
print("==========================================================================")
