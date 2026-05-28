import numpy as np
import cv2


def get_color_features(img, mask):

    #RGB mean and std inside the lesion.
    # cv2 loads images as BGR not RGB!!

    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0

    img_rgb = img[:, :, ::-1]   # BGR to RGB
    
    lesion_pixels = img_rgb[binary_mask]

    if len(lesion_pixels) == 0:
        return {
            "mean_r": 0, "mean_g": 0, "mean_b": 0,
            "std_r": 0, "std_g": 0, "std_b": 0,
        }

    return {
        "mean_r": np.mean(lesion_pixels[:, 0]),
        "mean_g": np.mean(lesion_pixels[:, 1]),
        "mean_b": np.mean(lesion_pixels[:, 2]),
        "std_r":  np.std(lesion_pixels[:, 0]),
        "std_g":  np.std(lesion_pixels[:, 1]),
        "std_b":  np.std(lesion_pixels[:, 2]),
    }
