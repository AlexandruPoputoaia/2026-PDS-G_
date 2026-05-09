import numpy as np
import cv2

def _circular_mean_std(hue_values):
    """
    Computes the circular mean and std of hue values.

    OpenCV hue is in [0, 179] representing [0°, 358°] (1 unit = 2°).
    A regular mean is wrong when hues wrap around (e.g. hues 2 and 178
    are both near red but would average to 90, which is green).

    :param hue_values: 1D numpy array of hue values in [0, 179]
    :return: (circular_mean, circular_std) in OpenCV hue units
    """
    # Convert to radians (1 hue unit = 2°)
    angles = hue_values * 2.0 * np.pi / 180.0

    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))

    # Circular mean — arctan2 gives angle in [-pi, pi], map back to [0, 180)
    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean_hue = (mean_angle * 180.0 / (2.0 * np.pi)) % 180.0

    # Circular std — derived from the mean resultant length R
    R = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
    R = np.clip(R, 0.0, 1.0)
    std_angle = np.sqrt(-2.0 * np.log(R + 1e-10))  # radians
    std_hue = std_angle * 180.0 / (2.0 * np.pi)

    return mean_hue, std_hue


def get_hsv_features(img, mask):
    """
    Calculates color features in HSV color space inside the lesion.
    HSV is often more informative than RGB for skin lesions because:
    - Hue captures the actual color
    - Saturation captures color intensity
    - Value captures brightness

    Hue uses circular statistics to handle wrap-around correctly.

    :param img: numpy array of the image (BGR format from cv2)
    :param mask: numpy array of the mask (grayscale)
    :return: dictionary with HSV statistics
    """
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    binary_mask = mask > 0

    # Convert BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract pixels inside lesion
    lesion_pixels = img_hsv[binary_mask]

    if len(lesion_pixels) == 0:
        return {
            "mean_h": 0, "mean_s": 0, "mean_v": 0,
            "std_h": 0, "std_s": 0, "std_v": 0
        }

    mean_h, std_h = _circular_mean_std(lesion_pixels[:, 0].astype(np.float64))

    return {
        "mean_h": mean_h,
        "mean_s": np.mean(lesion_pixels[:, 1]),
        "mean_v": np.mean(lesion_pixels[:, 2]),
        "std_h": std_h,
        "std_s": np.std(lesion_pixels[:, 1]),
        "std_v": np.std(lesion_pixels[:, 2]),
    }