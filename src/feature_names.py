# all feature column names in the same order as features.csv
# if you add a new feature add it here too

# shape features
SHAPE_FEATURES = ["area", "height", "width", "perimeter", "compactness", "solidity", "extent"]

# the ABC of ABCD (Asymmetry, Border, Color, Diameter)
ASYMMETRY_FEATURES = ["asymmetry"]
BORDER_FEATURES = ["border_irregularity", "border_gradient"]

# diameters at 4 angles + the elongation ratio
DIAMETER_FEATURES = ["diameter_0", "diameter_45", "diameter_90", "diameter_135", "diameter_ratio"]

# color stuff -- rgb, hsv, relative to surrounding skin
COLOR_RGB_FEATURES = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
COLOR_HSV_FEATURES = ["mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v"]
RELATIVE_COLOR_FEATURES = ["rel_r", "rel_g", "rel_b"]

# hue histogram - 8 bins
HUE_HIST_FEATURES = [f"hue_hist_{i}" for i in range(8)]

# texture
GLCM_FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
LBP_FEATURES = [f"lbp_{i}" for i in range(10)]   # P=8 uniform -> 10 bins

# hair
HAIR_FEATURES = ["hair_coverage", "hair_in_lesion"]


# the full list (ORDER MATTERS - matches features.csv)
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
