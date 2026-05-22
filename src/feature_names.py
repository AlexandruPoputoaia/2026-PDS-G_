"""
Single source of truth for the ordered list of feature columns.

Keep this in sync with what `src.extract_features` writes to features.csv.
Any code that consumes features.csv (main.py, plot_results.py,
open_question.py) imports FEATURE_COLS from here so feature additions
only need to happen in one place.
"""

# Shape
SHAPE_FEATURES = [
    "area", "height", "width", "perimeter", "compactness",
    "solidity", "extent",
]

# Asymmetry
ASYMMETRY_FEATURES = ["asymmetry"]

# Border
BORDER_FEATURES = ["border_irregularity", "border_gradient"]

# Diameter
DIAMETER_FEATURES = [
    "diameter_0", "diameter_45", "diameter_90", "diameter_135",
    "diameter_ratio",
]

# Color — summary stats
COLOR_RGB_FEATURES = [
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
]
COLOR_HSV_FEATURES = [
    "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
]
RELATIVE_COLOR_FEATURES = ["rel_r", "rel_g", "rel_b"]

# Color — hue histogram (8 bins)
HUE_HIST_FEATURES = [f"hue_hist_{i}" for i in range(8)]

# Texture — GLCM
GLCM_FEATURES = [
    "contrast", "dissimilarity", "homogeneity", "energy", "correlation",
]

# Texture — LBP (P=8, uniform → 10 bins)
LBP_FEATURES = [f"lbp_{i}" for i in range(10)]

# Hair
HAIR_FEATURES = ["hair_coverage", "hair_in_lesion"]


# Concatenated in the order they appear in features.csv
FEATURE_COLS = (
    SHAPE_FEATURES
    + ASYMMETRY_FEATURES
    + BORDER_FEATURES
    + DIAMETER_FEATURES
    + COLOR_RGB_FEATURES
    + COLOR_HSV_FEATURES
    + HUE_HIST_FEATURES
    + RELATIVE_COLOR_FEATURES
    + GLCM_FEATURES
    + LBP_FEATURES
    + HAIR_FEATURES
)
