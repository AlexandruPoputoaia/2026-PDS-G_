import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils import find_image, find_mask
from src.feature_area import get_lesion_area
from src.feature_shape import (get_lesion_dimensions, get_perimeter,
                                get_compactness, get_solidity, get_extent)
from src.feature_color import get_color_features
from src.feature_color_hsv import get_hsv_features
from src.feature_color_histogram import get_color_histogram
from src.feature_relative_color import get_relative_color
from src.feature_asymmetry import get_asymmetry
from src.feature_border import get_border_irregularity, get_border_gradient
from src.feature_diameter import get_all_diameters
from src.feature_texture import get_texture_features
from src.feature_lbp import get_lbp_features
from src.feature_hair import get_hair_features


def extract_features_for_image(img_id):
    """
    Extracts all features for a single image.

    :param img_id: image filename (e.g. PAT_1516_1765_530.png)
    :return: dictionary of features, or None if image/mask not found
    """
    # Find image and mask
    img_path = find_image(img_id)
    mask_path = find_mask(img_id)

    if img_path is None or mask_path is None:
        return None

    # Load image and mask
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        return None

    features = {}
    features["img_id"] = img_id

    # Shape features
    features["area"] = get_lesion_area(mask, img)
    height, width = get_lesion_dimensions(mask, img)
    features["height"] = height
    features["width"] = width
    features["perimeter"] = get_perimeter(mask, img)
    features["compactness"] = get_compactness(mask, img)
    features["solidity"] = get_solidity(mask, img)
    features["extent"] = get_extent(mask, img)

    # Asymmetry
    features["asymmetry"] = get_asymmetry(mask, img)

    # Border features
    features["border_irregularity"] = get_border_irregularity(mask, img)
    features["border_gradient"] = get_border_gradient(mask, img)

    # Diameter features
    diameters = get_all_diameters(mask, img)
    features.update(diameters)

    # RGB color features
    color = get_color_features(img, mask)
    features.update(color)

    # HSV color features (summary statistics)
    hsv = get_hsv_features(img, mask)
    features.update(hsv)

    # Hue histogram (captures multi-modal color distributions)
    hue_hist = get_color_histogram(img, mask)
    features.update(hue_hist)

    # Relative color features
    rel_color = get_relative_color(img, mask)
    features.update(rel_color)

    # Texture features (GLCM)
    texture = get_texture_features(img, mask)
    features.update(texture)

    # LBP texture features
    lbp = get_lbp_features(img, mask)
    features.update(lbp)

    # Hair features
    hair = get_hair_features(img, mask)
    features.update(hair)

    return features


def main():
    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv("data/metadata.csv")
    print(f"Total images: {len(df)}")

    # Loop over all images and extract features
    print("Extracting features...")
    results = []
    failed = []

    for img_id in tqdm(df["img_id"]):
        features = extract_features_for_image(img_id)
        if features is not None:
            results.append(features)
        else:
            failed.append(img_id)

    print(f"Successfully processed: {len(results)} images")
    if failed:
        print(f"Failed ({len(failed)} images): {failed}")

    # Convert to dataframe
    features_df = pd.DataFrame(results)

    # Add diagnostic label and patient_id from metadata
    features_df = features_df.merge(
        df[["img_id", "diagnostic", "patient_id"]],
        on="img_id",
        how="left"
    )

    # Save to CSV
    output_path = Path("data/features.csv")
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    print(f"Features: {list(features_df.columns)}")
    print(f"Total features: {len(features_df.columns) - 3}")  # minus img_id, diagnostic, patient_id


if __name__ == "__main__":
    main()