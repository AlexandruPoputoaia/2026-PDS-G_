from pathlib import Path
import os

# data paths
IMG_DIR = Path("data/imgs")
MASK_DIR = Path("data/masks")

# NOTE: we used to have 3 subfolders (imgs_part_1, imgs_part_2, imgs_part_3)
# but it was confusing so we flattened everything into one. If something breaks
# with paths it's probably this.
# OLD_IMG_DIRS = ["data/imgs_part_1", "data/imgs_part_2", "data/imgs_part_3"]


def find_image(img_id):
    """Look for the image file given its id (e.g. PAT_1516_1765_530.png)."""
    path = IMG_DIR / img_id
    if path.exists():
        return path
    # print(f"missing image: {img_id}")   # uncomment for debugging
    return None


def find_mask(img_id):
    # mask file is named like <img_stem>_mask.png
    stem = Path(img_id).stem
    mask_name = stem + "_mask.png"
    path = MASK_DIR / mask_name
    if path.exists():
        return path
    return None
