import numpy as np
import cv2


def circular_mean_std(hue_values):
    # normal mean is wrong for hue because it wraps around (0-179 in opencv)
    # e.g. hue 2 and 178 are both reddish but np.mean gives 90 which is green
    # found a fix on stackoverflow using sin/cos to handle the wraparound
    # opencv hue goes 0-179, so 1 unit = 2 degrees
    angles = hue_values * 2.0 * np.pi / 180.0

    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))

    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean_hue = (mean_angle * 180.0 / (2.0 * np.pi)) % 180.0

    # std using mean resultant length (also from that post)
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    R = np.clip(R, 0.0, 1.0)
    std_angle = np.sqrt(-2.0 * np.log(R + 1e-10))
    std_hue = std_angle * 180.0 / (2.0 * np.pi)

    return mean_hue, std_hue


def get_hsv_features(img, mask):
    """HSV color features inside the lesion (mean and std of H, S, V)."""
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lesion_pixels = img_hsv[binary_mask]

    if len(lesion_pixels) == 0:
        return {
            "mean_h": 0, "mean_s": 0, "mean_v": 0,
            "std_h": 0, "std_s": 0, "std_v": 0,
        }

    mean_h, std_h = circular_mean_std(lesion_pixels[:, 0].astype(np.float64))

    return {
        "mean_h": mean_h,
        "mean_s": np.mean(lesion_pixels[:, 1]),
        "mean_v": np.mean(lesion_pixels[:, 2]),
        "std_h":  std_h,
        "std_s":  np.std(lesion_pixels[:, 1]),
        "std_v":  np.std(lesion_pixels[:, 2]),
    }
