# dataset.py
# loads the COCO-style MAGFiLO annotations and creates a PyTorch Dataset
# for solar filament segmentation.

import json
import os
import random
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# import config from the same folder
import config


def set_seed(seed: int = config.SEED):
    """fix randomness across the libraries we use."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_coco_annotations(json_path: str):
    """read the COCO json and return the whole dictionary."""
    with open(json_path, "r") as f:
        return json.load(f)


def group_annotations_by_filename(coco: dict, remove_ambiguous: bool = True):
    """
    the COCO json has 1154 image entries but only 707 actual jpg files.
    some files are annotated by multiple people, so they have multiple image_ids.
    this function groups all annotations by physical filename.

    returns:
        file_to_imgids: dict[filename] -> list of image_ids
        file_to_anns:   dict[filename] -> list of annotation dicts
    """
    filename_to_ids = defaultdict(list)
    for img in coco["images"]:
        filename_to_ids[img["file_name"]].append(img["id"])

    # collect annotations per image_id first
    id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        if remove_ambiguous and ann.get("category_id") == 4:
            continue
        id_to_anns[ann["image_id"]].append(ann)

    file_to_imgids = dict(filename_to_ids)
    file_to_anns = {}
    for fname, imgids in file_to_imgids.items():
        anns = []
        for iid in imgids:
            anns.extend(id_to_anns[iid])
        file_to_anns[fname] = anns

    return file_to_imgids, file_to_anns


def polygon_to_mask(polygon: List[float], h: int, w: int) -> np.ndarray:
    """draw a closed polygon (flat list of x,y,x,y,...) as a binary mask."""
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    # make sure the polygon is closed
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    return mask


def build_unified_mask(annotations: List[dict], h: int = 2048, w: int = 2048) -> np.ndarray:
    """
    combine every polygon in the annotation list into one binary mask.
    overlapping filaments all become 1 (semantic mask).
    """
    mask = np.zeros((h, w), dtype=np.float32)
    for ann in annotations:
        # segmentation is a list of one or more polygons
        segs = ann["segmentation"]
        if isinstance(segs, list):
            for seg in segs:
                if len(seg) >= 6:
                    m = polygon_to_mask(seg, h, w)
                    mask = np.maximum(mask, m)
    return mask


def get_train_val_files(file_to_imgids: dict, n_folds: int = 5, val_fold: int = 0, seed: int = 2026):
    """
    split the list of physical filenames into train and val.
    the split is by file, not by image_id, to avoid leakage.
    group = year from filename (positions 0-4) so all images of one year
    stay in the same fold.
    """
    from sklearn.model_selection import GroupKFold

    filenames = sorted(file_to_imgids.keys())
    groups = [fn[:4] for fn in filenames]

    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(filenames, [0] * len(filenames), groups))
    train_idx, val_idx = splits[val_fold]

    train_files = [filenames[i] for i in train_idx]
    val_files = [filenames[i] for i in val_idx]
    return train_files, val_files


def split_by_year(file_to_imgids: dict, train_years=range(2011, 2020), val_years=range(2020, 2023)):
    """alternative: hold out recent years as validation."""
    train_files = []
    val_files = []
    for fn in sorted(file_to_imgids.keys()):
        year = int(fn[:4])
        if year in train_years:
            train_files.append(fn)
        elif year in val_years:
            val_files.append(fn)
        else:
            # any other year goes to train
            train_files.append(fn)
    return train_files, val_files


def get_augmentations(image_size: int = config.TRAIN_RES, is_train: bool = True):
    """Albumentations pipeline; tensor conversion is always last."""
    ops = []
    if is_train:
        ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.8, 1.2), translate_percent=(-0.05, 0.05), rotate=(-30, 30), p=0.5)]
        if config.HEAVY_AUG:
            distortions = [A.ElasticTransform(alpha=40, sigma=8, p=1), A.GridDistortion(p=1)]
            ops += [A.OneOf(distortions, p=0.6), A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.4), A.ISONoise(p=0.25),
                    A.GaussNoise(std_range=(0.01, 0.06), p=0.3), A.CLAHE(p=0.4)]
            grid = getattr(A, "GridDropout", None) or getattr(A, "GridMask", None)
            if grid is not None:
                ops.append(grid(p=0.25))
    else:
        ops.append(A.Resize(image_size, image_size))
    channels = config.IN_CHANNELS
    ops += [A.Normalize(mean=(0.5,) * channels, std=(0.5,) * channels), ToTensorV2()]
    return A.Compose(ops)


class FilamentDataset(Dataset):
    """
    PyTorch dataset for solar filament segmentation.

    for each physical jpg file we load the 2048x2048 image once,
    rasterize all annotations into a binary mask, then either
    random-crop a 1024x1024 region for training or resize to 1024 for validation.
    """

    def __init__(
        self,
        file_to_anns: dict,
        train_files: List[str],
        image_dir: str,
        image_size: int = config.TRAIN_RES,
        is_train: bool = True,
        seed: int = config.SEED,
    ):
        self.file_to_anns = file_to_anns
        self.image_dir = image_dir
        self.image_size = image_size
        self.is_train = is_train
        self.filenames = sorted(train_files)
        self.rng = np.random.default_rng(seed)

        # load all masks into memory once (707 images x 2048 x 2048 x 4 bytes ~ 11 GB, too much)
        # instead we only cache image id and annotation list, and load image on the fly.
        # if you have >16 GB ram you can cache the masks to speed training.
        self.transform = get_augmentations(image_size, is_train)

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, fname: str) -> np.ndarray:
        path = os.path.join(self.image_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"image not found: {path}")
        return img.astype(np.float32) / 255.0

    def _load_mask(self, fname: str) -> np.ndarray:
        anns = self.file_to_anns[fname]
        if not anns:
            return np.zeros((2048, 2048), dtype=np.float32)
        return build_unified_mask(anns, 2048, 2048)

    def _sample_crop(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        for training, crop a square patch of size image_size from the 2048 image.
        with 75% probability the crop is centered on a random filament pixel,
        otherwise it is a random crop anywhere on the disk.
        """
        h, w = img.shape
        crop_h = crop_w = self.image_size

        # find foreground pixels
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and self.rng.random() < 0.75:
            idx = self.rng.integers(0, len(xs))
            cx, cy = xs[idx], ys[idx]
        else:
            cx = self.rng.integers(0, w)
            cy = self.rng.integers(0, h)

        # top-left corner
        x1 = min(max(cx - crop_w // 2, 0), w - crop_w)
        y1 = min(max(cy - crop_h // 2, 0), h - crop_h)

        return img[y1 : y1 + crop_h, x1 : x1 + crop_w], mask[y1 : y1 + crop_h, x1 : x1 + crop_w]

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        img = self._load_image(fname)
        mask = self._load_mask(fname)

        if self.is_train and config.MASK_AWARE_CROP:
            img, mask = self._sample_crop(img, mask)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if config.IN_CHANNELS == 3 and img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        sample = self.transform(image=img, mask=mask)
        img_tensor = sample["image"].float()
        mask_tensor = sample["mask"].float()
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return img_tensor, mask_tensor, os.path.splitext(fname)[0]


def make_dataloaders(
    coco_json_path: str = config.TRAIN_JSON,
    image_dir: str = config.TRAIN_IMAGES,
    n_folds: int = config.N_FOLDS,
    val_fold: int = config.VAL_FOLD,
    image_size: int = config.TRAIN_RES,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    split_mode: str = "group_kfold",  # or "year"
):
    """build train and validation dataloaders."""
    coco = load_coco_annotations(coco_json_path)
    _, file_to_anns = group_annotations_by_filename(coco)

    if split_mode == "group_kfold":
        train_files, val_files = get_train_val_files(
            dict.fromkeys(file_to_anns.keys()), n_folds, val_fold
        )
    elif split_mode == "year":
        train_files, val_files = split_by_year(
            dict.fromkeys(file_to_anns.keys())
        )
    else:
        # random 85/15 by file
        all_files = sorted(file_to_anns.keys())
        from sklearn.model_selection import train_test_split
        train_files, val_files = train_test_split(
            all_files, test_size=0.15, random_state=config.SEED
        )

    train_ds = FilamentDataset(
        file_to_anns, train_files, image_dir, image_size, is_train=True
    )
    val_ds = FilamentDataset(
        file_to_anns, val_files, image_dir, image_size, is_train=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=lambda x: set_seed(config.SEED + x),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(f"train images: {len(train_ds)} | val images: {len(val_ds)}")
    return train_loader, val_loader, file_to_anns, val_files


if __name__ == "__main__":
    # quick sanity check
    set_seed()
    train_loader, val_loader, file_to_anns, val_files = make_dataloaders(num_workers=0)
    for batch in train_loader:
        img, mask, base_id = batch
        print("batch img shape:", img.shape, "mask shape:", mask.shape, "sample:", base_id[0])
        break
