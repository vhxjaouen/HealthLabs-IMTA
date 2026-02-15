import os, json
import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    LoadImaged, SqueezeDimd, Transposed,
    Resized, ScaleIntensityRangePercentilesd, RandFlipd, RandRotate90d, EnsureChannelFirstd, CenterSpatialCropd,
    EnsureTyped, Compose, MapTransform,SpatialPadd
)


def load_split_from_json(data_dir, split="training"):
    with open(os.path.join(data_dir, "dataset.json")) as f:
        dataset = json.load(f)
    items = []
    for entry in dataset[split]:
        img = os.path.join(data_dir, entry["image"])  
        lbl = os.path.join(data_dir, entry["label"])
        items.append({"image": img, "label": lbl})
    return items

def get_seg2d_dataloaders(
    data_dir,
    batch_size=8,
    num_workers=2,
    cache_rate=0.1,
    val_fraction=0.2,
    seed=42,
    target_size=(256, 256)
):
    all_train = load_split_from_json(data_dir, "training")
    np.random.seed(seed)
    np.random.shuffle(all_train)
    val_size = int(len(all_train) * val_fraction)
    val_files, train_files = all_train[:val_size], all_train[val_size:]

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        Transposed(keys="image", indices=(2, 0, 1)),
        SqueezeDimd(keys="label", dim=-1),
        EnsureChannelFirstd(keys="label"),
        SpatialPadd(keys=["image", "label"], spatial_size=target_size),
        CenterSpatialCropd(keys=["image","label"], roi_size=target_size), # si dataset plus grand
        ScaleIntensityRangePercentilesd(
            keys="image",
            lower=1, upper=99,
            b_min=-1.0, b_max=1.0,
            clip=True
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        Transposed(keys="image", indices=(2, 0, 1)),
        SqueezeDimd(keys="label", dim=-1),
        EnsureChannelFirstd(keys="label"),
        SpatialPadd(keys=["image", "label"], spatial_size=target_size),
        CenterSpatialCropd(keys=["image","label"], roi_size=target_size), # si dataset plus grand
        ScaleIntensityRangePercentilesd(
            keys="image",
            lower=1, upper=99,
            b_min=-1.0, b_max=1.0,
            clip=True
        ),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(train_files, train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    val_ds   = CacheDataset(val_files, val_transforms,   cache_rate=cache_rate, num_workers=num_workers)

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)


# --- Expand dataset into per-modality samples ---
def expand_modalities(files, modality_names=("t1ce", "t2", "flair")):
    expanded = []
    for f in files:  # each f = {"image": "path/to/nii.gz", "label": ...}
        img_path = f["image"]
        for i, name in enumerate(modality_names):
            expanded.append({
                "image": img_path,
                "modality_idx": i,  # which channel to select
                "label": i          # classification target
            })
    return expanded


# --- Transform: pick one channel ---
class SelectModalityd(MapTransform):
    def __init__(self, keys, modality_key="modality_idx"):
        super().__init__(keys)
        self.modality_key = modality_key

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]  # shape (H,W,3)
        i = d[self.modality_key]
        d["image"] = img[..., i][None, ...]  # (1,H,W)
        d["label"] = int(i)
        return d


# --- Classification dataloaders ---
def get_classif_dataloaders(
    data_dir,
    batch_size=16,
    num_workers=2,
    cache_rate=1.0,
    target_size=(256, 256),
    val_fraction=0.1,
    seed=42
):
    from utils.data_utils import load_split_from_json  # reuse your helper

    np.random.seed(seed)
    all_files = load_split_from_json(data_dir, "training")
    np.random.shuffle(all_files)

    val_size = int(len(all_files) * val_fraction)
    val_files, train_files = all_files[:val_size], all_files[val_size:]

    # expand each image into 3 classification samples
    train_files = expand_modalities(train_files)
    val_files   = expand_modalities(val_files)

    transforms = Compose([
        LoadImaged(keys="image"),
        SelectModalityd(keys="image"),
        Resized(keys="image", spatial_size=target_size, mode="bilinear"),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=1, upper=99, b_min=-1.0, b_max=1.0, clip=True
        ),
        EnsureTyped(keys=("image", "label")),
    ])

    train_ds = CacheDataset(data=train_files, transform=transforms,
                            cache_rate=cache_rate, num_workers=num_workers)
    val_ds   = CacheDataset(data=val_files, transform=transforms,
                            cache_rate=cache_rate, num_workers=num_workers)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=1,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# pour I2I  T1ce → T2
class T1ce2T2d(MapTransform):
    """
    Transform: prend une image (H,W,3) et retourne
    un dict {"image": T1ce, "label": T2}.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]  # (H,W,3)
        t1ce = img[...,0][None,...]  # (1,H,W)
        t2   = img[...,1][None,...]  # (1,H,W)
        return {"image": t1ce, "label": t2}

def get_i2i_dataloaders(
    data_dir,
    batch_size=8,
    num_workers=2,
    cache_rate=0.1,
    val_fraction=0.2,
    seed=42,
    target_size=(256, 256)
):
    all_train = load_split_from_json(data_dir, "training")
    np.random.seed(seed)
    np.random.shuffle(all_train)
    val_size = int(len(all_train) * val_fraction)
    val_files, train_files = all_train[:val_size], all_train[val_size:]

    train_transforms = Compose([
        LoadImaged(keys="image"),
        T1ce2T2d(keys="image"),   # génère "image"=T1ce, "label"=T2
        SpatialPadd(keys=["image", "label"], spatial_size=target_size),
        CenterSpatialCropd(keys=["image","label"], roi_size=target_size), # si dataset plus grand
        ScaleIntensityRangePercentilesd(
            keys=["image", "label"],
            lower=1, upper=99,
            b_min=-1.0, b_max=1.0,
            clip=True
        ),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys="image"),
        T1ce2T2d(keys="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=target_size),
        CenterSpatialCropd(keys=["image","label"], roi_size=target_size), # si dataset plus grand  
        ScaleIntensityRangePercentilesd(
            keys=["image", "label"],
            lower=1, upper=99,
            b_min=-1.0, b_max=1.0,
            clip=True
        ),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(train_files, train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    val_ds   = CacheDataset(val_files, val_transforms,   cache_rate=cache_rate, num_workers=num_workers)

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

