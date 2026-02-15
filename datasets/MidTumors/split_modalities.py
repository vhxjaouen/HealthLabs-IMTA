import os
import glob
import nibabel as nib
import numpy as np
import shutil

def split_modalities():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    trainA_dir = os.path.join(base_dir, 'trainA')
    trainB_dir = os.path.join(base_dir, 'trainB')
    
    # New dataset names
    new_datasets = {
        't1ce': 'MidTumors_t1ce',
        't2n': 'MidTumors_t2n', 
        't2f': 'MidTumors_t2f'
    }
    
    # Map modality name to slice index
    modality_indices = {
        't1ce': 0,
        't2n': 1,
        't2f': 2
    }
    
    # Get parent dir of current dataset to place new datasets alongside
    parent_dir = os.path.dirname(base_dir)
    
    # Check output directories
    out_dirs = {}
    for mod, name in new_datasets.items():
        path = os.path.join(parent_dir, name)
        out_dirs[mod] = path
        os.makedirs(os.path.join(path, 'trainA'), exist_ok=True)
        os.makedirs(os.path.join(path, 'trainB'), exist_ok=True)

    # Get file list
    files = sorted(glob.glob(os.path.join(trainA_dir, '*_3c.nii.gz')))
    print(f"Found {len(files)} files to process.")

    for f_path in files:
        fname = os.path.basename(f_path)
        # ID is everything before _3c
        id_str = fname.replace('_3c.nii.gz', '')
        
        # Corresponding segmentation file
        seg_fname = f"{id_str}_seg.nii.gz"
        seg_path = os.path.join(trainB_dir, seg_fname)
        
        if not os.path.exists(seg_path):
            print(f"Skipping {fname}: Segmentation file not found: {seg_path}")
            continue
            
        print(f"Processing {id_str}...")

        # Load images
        img_nii = nib.load(f_path)
        seg_nii = nib.load(seg_path)
        
        img_data = img_nii.get_fdata()
        
        # Verify shape
        if img_data.ndim != 3 or img_data.shape[2] != 3:
            print(f"  Warning: Unexpected shape {img_data.shape} for {fname}. Skipping.")
            continue
            
        target_affine = seg_nii.affine
        target_header = seg_nii.header
        
        # Process each modality
        for mod, idx in modality_indices.items():
            # Extract slice
            slice_data = img_data[:, :, idx]
            
            # Reshape to (H, W, 1) to match segmentation if it is 3D (H,W,1)
            # The segmentation was (240, 240, 1) in our check.
            if slice_data.ndim == 2:
                slice_data = slice_data[:, :, np.newaxis]
            
            # Create new image with seg header
            new_img = nib.Nifti1Image(slice_data, target_affine, header=target_header)
            
            # Output filename: use simple ID.nii.gz for both image and seg
            # to accommodate pix2pix/cyclegan conventions (often paired by name)
            # Or should we keep original names?
            # User wants "simpler one... take the header of the trainB seg".
            # Let's use {id_str}.nii.gz for simplicity and consistency.
            out_name = f"{id_str}.nii.gz"
            
            # Save Image to trainA
            out_img_path = os.path.join(out_dirs[mod], 'trainA', out_name)
            nib.save(new_img, out_img_path)
            
            # Save Seg to trainB
            # We can just copy the seg file to the new name/location, but since we want to ensure
            # header matches perfectly (we used seg header for image), saving the loaded seg object is safest.
            out_seg_path = os.path.join(out_dirs[mod], 'trainB', out_name)
            nib.save(seg_nii, out_seg_path)
            
    print("Done!")

if __name__ == "__main__":
    split_modalities()
