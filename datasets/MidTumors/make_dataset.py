import os, glob, json

def make_dataset_json(root_dir=None, output="dataset.json"):
    """
    Generate a dataset.json for 2D BraTS-style data:
    - trainA/: *_3c.nii.gz (3-channel inputs)
    - trainB/: *_seg.nii.gz (multi-class segmentation masks)
    """
    # Default: dataset root = directory of this Python file
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))

    images = sorted(glob.glob(os.path.join(root_dir, "trainA", "*_3c.nii.gz")))
    labels = sorted(glob.glob(os.path.join(root_dir, "trainB", "*_seg.nii.gz")))

    # Build dicts { patient_id : filepath }
    img_dict = {os.path.basename(f).replace("_3c.nii.gz", ""): f for f in images}
    lbl_dict = {os.path.basename(f).replace("_seg.nii.gz", ""): f for f in labels}

    training = []
    for pid, img_path in img_dict.items():
        if pid in lbl_dict:
            lbl_path = lbl_dict[pid]
            training.append({
                # prefix with "./" so paths are portable and join cleanly
                "image": "./" + os.path.relpath(img_path, root_dir),
                "label": "./" + os.path.relpath(lbl_path, root_dir)
            })
        else:
            print(f"⚠️ No label found for {pid}, skipping")

    dataset = {
        "name": "BraTS2D_MidSlices",
        "description": "BraTS mid-slice dataset with 3 modalities stacked as channels",
        "tensorImageSize": "2D",
        "modality": {"0": "t1ce", "1": "t2", "2": "flair"},
        "labels": {
            "0": "background",
            "1": "edema",
            "2": "non-enhancing",
            "3": "enhancing"
        },
        "numTraining": len(training),
        "training": training
    }

    out_path = os.path.join(root_dir, output)
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Wrote {out_path} with {len(training)} training samples.")


if __name__ == "__main__":
    make_dataset_json()
