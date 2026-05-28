import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils import find_image, find_mask
from src.feature_area import get_lesion_area
from src.feature_shape import get_lesion_dimensions, get_perimeter, get_compactness, get_solidity, get_extent
from src.feature_asymmetry import get_asymmetry
from src.feature_border import get_border_irregularity, get_border_gradient
from src.feature_diameter import get_all_diameters
from src.feature_color import get_color_features
from src.feature_color_hsv import get_hsv_features
from src.feature_color_histogram import get_color_histogram
from src.feature_relative_color import get_relative_color
from src.feature_texture import get_texture_features
from src.feature_lbp import get_lbp_features
from src.feature_hair import get_hair_features


def extract_features_for_image(img_id):
    """Extract all features for one image. Returns None if image or mask missing."""
    img_path = find_image(img_id)
    mask_path = find_mask(img_id)

    if img_path is None or mask_path is None:
        return None

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        return None

    features = {"img_id": img_id}

    # ---- shape ----
    features["area"] = get_lesion_area(mask, img)
    h, w = get_lesion_dimensions(mask, img)
    features["height"] = h
    features["width"] = w
    features["perimeter"] = get_perimeter(mask, img)
    features["compactness"] = get_compactness(mask, img)
    features["solidity"] = get_solidity(mask, img)
    features["extent"] = get_extent(mask, img)

    # ---- asymmetry ----
    features["asymmetry"] = get_asymmetry(mask, img)

    # ---- border ----
    features["border_irregularity"] = get_border_irregularity(mask, img)
    features["border_gradient"] = get_border_gradient(mask, img)

    # ---- diameter ----
    features.update(get_all_diameters(mask, img))

    # ---- color ----
    features.update(get_color_features(img, mask))
    features.update(get_hsv_features(img, mask))
    features.update(get_color_histogram(img, mask))
    features.update(get_relative_color(img, mask))

    # ---- texture ----
    features.update(get_texture_features(img, mask))
    features.update(get_lbp_features(img, mask))

    # ---- hair ----
    features.update(get_hair_features(img, mask))

    return features


def main():
    print("Loading metadata...")
    df = pd.read_csv("data/metadata.csv")
    print(f"Total images in metadata: {len(df)}")

    print("Extracting features (this takes a while)...")
    results = []
    failed = []

    for img_id in tqdm(df["img_id"], desc="features"):
        feats = extract_features_for_image(img_id)
        if feats is not None:
            results.append(feats)
        else:
            failed.append(img_id)

    print(f"Done. Processed {len(results)} images, failed on {len(failed)}.")
    if failed:
        # only print first 10 so the terminal doesn't explode
        print("First failed ids:", failed[:10])

    features_df = pd.DataFrame(results)

    # merge in the diagnostic label and the patient_id so they're in features.csv too
    features_df = features_df.merge(
        df[["img_id", "diagnostic", "patient_id"]],
        on="img_id", how="left",
    )

    output_path = Path("data/features.csv")
    features_df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}")
    print(f"Columns: {list(features_df.columns)}")
    print(f"Number of feature columns: {len(features_df.columns) - 3}")  # excludes img_id, diagnostic, patient_id


if __name__ == "__main__":
    main()
