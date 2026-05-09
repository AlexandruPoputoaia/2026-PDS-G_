import numpy as np
import cv2

def get_color_features(img, mask):
    """
    Calculates mean and standard deviation of R, G, B channels
    inside the lesion region only.
    
    :param img: numpy array of the image (BGR format from cv2)
    :param mask: numpy array of the mask (grayscale)
    :return: dictionary with mean and std for each channel
    """
    # Make sure mask and image have the same size
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask > 0
    
    # cv2 loads images as BGR, convert to RGB
    img_rgb = img[:, :, ::-1]
    
    # Extract only the pixels inside the lesion
    lesion_pixels = img_rgb[binary_mask]
    
    if len(lesion_pixels) == 0:
        return {
            "mean_r": 0, "mean_g": 0, "mean_b": 0,
            "std_r": 0, "std_g": 0, "std_b": 0
        }
    
    return {
        "mean_r": np.mean(lesion_pixels[:, 0]),
        "mean_g": np.mean(lesion_pixels[:, 1]),
        "mean_b": np.mean(lesion_pixels[:, 2]),
        "std_r": np.std(lesion_pixels[:, 0]),
        "std_g": np.std(lesion_pixels[:, 1]),
        "std_b": np.std(lesion_pixels[:, 2]),
    }