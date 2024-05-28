from PIL import Image
import numpy as np

def are_same_dimensions(image1_path, image2_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    # Get the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size
    
    np_im1 = np.array(image1)
    np_im2 = np.array(image2)
    print(np_im1.shape)
    print(np_im2.shape)
    
    # Check if the dimensions are the same
    return (width1, height1) == (width2, height2)

# Example usage
image1_path = "./photos/woman_snow/000000000785.jpg"
image2_path = "./photos/woman_snow/best_segment.png"
if are_same_dimensions(image1_path, image2_path):
    print("The images have the same dimensions.")
else:
    print("The images do not have the same dimensions.")
