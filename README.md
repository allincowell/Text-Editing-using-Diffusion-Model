# Text-based Image Editing using Diffusion Model and CLIP

This repository contains the implementation of a state-of-the-art text-guided image editing pipeline. The project leverages advanced machine learning techniques, including Segformer, CLIP, and diffusion models, to allow users to specify image edits through textual descriptions. The approach preserves the original context and background of the images, providing high-quality and coherent edits.

## Features

- **End-to-End Automated Pipeline:** A seamless system enabling users to edit images by simply describing the desired changes in text.
- **Advanced Image Segmentation:** Utilizes a pretrained Segformer model for precise segmentation of the target object in the image.
- **Optimal Mask Identification:** Employs CLIP to find the best mask that aligns with the textual description, ensuring accurate and relevant edits.
- **Text-Guided Image Editing:** Integrates a diffusion model in the denoising step, guided by CLIP, to implement the specified edits while maintaining the image’s background context.
- **Background Preservation:** Ensures that the edited images retain their original visual coherence through L-2 and LPIPS loss functions.

## How It Works

1. **Image Segmentation:** The Segformer model identifies and segments the object to be edited in the input image.
2. **Mask Selection:** CLIP evaluates multiple masks to find the one that best matches the textual description provided by the user.
3. **Diffusion Model Editing:** A diffusion model, guided by CLIP, applies the specified edits during the denoising step, integrating the changes seamlessly while preserving the original background.

## Comparative Analysis

The proposed method has been validated against existing models, such as DALLE2, and has demonstrated superior performance in maintaining background integrity and overall image quality.

## Hyperparameter Tuning

Detailed ablation studies and hyperparameter tuning were conducted to optimize model performance, identifying the optimal settings for λCLIP, λLPIPS, and λ2.

## Results and Impact

This innovative approach not only simplifies the image editing process but also opens new avenues for creative expression and data augmentation. The methodology and results have been documented comprehensively, contributing to the academic and professional discourse in automated image editing techniques.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers
- OpenCV
- CLIP
- Other dependencies listed in `requirements.txt`

### Installation

Clone this repository:
`git clone https://github.com/yourusername/text-based-image-editing.git`
`cd text-based-image-editing`

Install the required dependencies:
`pip install -r requirements.txt`

### Usage

Run the pipeline with your own images and text descriptions:
`python main.py --image_path /path/to/image --description "A description of the desired edit"`

You can directly use the run.sh file and make edits in the variables: `IMAGE_PATH, OUTPUT_DIR, SEGMENT_PROMPT` and `REPLACE_PROMPT`. Then use the command `bash run.sh`.

### Examples

Include examples of input images, text descriptions, and the resulting edited images to demonstrate the capabilities of the pipeline.

## Contributions

Feel free to contribute to this project by opening issues, submitting pull requests, or providing feedback.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
