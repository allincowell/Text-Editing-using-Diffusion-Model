import argparse
import os
import numpy as np
from PIL import Image
import cv2
import torch
import clip
from transformers import pipeline
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Sigmoid to ensure output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

# Define U-Net model
device = "cuda" if torch.cuda.is_available() else "cpu"
unet_model = UNet(in_channels=3, out_channels=1).to(device)
unet_model.load_state_dict(torch.load("path_to_trained_unet_model.pth"))
unet_model.eval()

# CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Text prompt preprocessing for CLIP
text_tokens = clip.tokenize([text_prompt]).to(device)

# Load and preprocess the original image
original_image = Image.open(image_name)
preprocessed_image = preprocess(original_image).unsqueeze(0).to(device)

# Segment the image using U-Net
with torch.no_grad():
    segment_masks = unet_model(preprocessed_image)

segment_masks = F.interpolate(segment_masks, size=original_image.size, mode='bilinear', align_corners=False)
segment_masks = (segment_masks > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)

# Apply masks and preprocess segmented images for CLIP
segment_image = []
for mask in segment_masks:
    mask_image = Image.fromarray(mask * 255)
    apply_mask(mask_image, output_dir)
    masked_image = Image.open(os.path.join(output_dir, 'mask.png'))
    segment_image.append(preprocess(masked_image))

segment_image_tensor = torch.stack(segment_image).to(device)

# Encode text and images using CLIP
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    image_features = clip_model.encode_image(segment_image_tensor)
    logits_per_image, logits_per_text = clip_model(image=segment_image_tensor, text=text_tokens)
    probs = logits_per_text.softmax(dim=-1).cpu().numpy()

# Save the best segment
best_mask_idx = np.argmax(probs)
best_mask = segment_masks[best_mask_idx]
best_mask_image = Image.fromarray(best_mask * 255)
best_mask_image.save(os.path.join(output_dir, 'best_segment.png'))

np.set_printoptions(precision=3, suppress=True)
print("==========================================================================")
print('The probability distribution over the segments are:')
print(probs)

print(f'The max. probability = {probs[0, best_mask_idx]}')
print("==========================================================================")
