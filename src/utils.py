from pathlib import Path
import os

# data paths
IMG_DIR = Path("data/imgs")
MASK_DIR = Path("data/masks")

# we used to have 3 subfolders (imgs_part_1, imgs_part_2, imgs_part_3)
# but it was confusing so we flattened everything into one.

def find_image(img_id):
    path = IMG_DIR / img_id
    if path.exists():
        return path
    # print(f"missing image: {img_id}")   
    return None


def find_mask(img_id):
    stem = Path(img_id).stem
    mask_name = stem + "_mask.png"
    path = MASK_DIR / mask_name
    if path.exists():
        return path
    # print(f"missing mask: {img_id}")
    return None
