import os
import glob
from PIL import Image
from tqdm import tqdm
import random

def prepare_dataset_lr(hr_dir, lr_dir, save_dir, patch_size=256, scale=4, patches_per_img=20):
    os.makedirs(os.path.join(save_dir, "HR"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "LR"), exist_ok=True)
    
    hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
    lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
    
    print(f"Found {len(hr_paths)} HR images and {len(lr_paths)} LR images.")
    
    patch_idx = 0
    for hr_path, lr_path in tqdm(zip(hr_paths, lr_paths), total=len(hr_paths)):
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")
        
        w, h = hr_img.size
        # Ensure LR and HR match in terms of scale
        # DIV2K LR is already downsampled. We need to upsample it to match HR size 
        # if the model expects same-size input (as in the current FreqUNet logic).
        lr_img_up = lr_img.resize((w, h), Image.BICUBIC)
        
        for _ in range(patches_per_img):
            # Random crop coordinates
            x = random.randint(0, w - patch_size)
            y = random.randint(0, h - patch_size)
            
            hr_patch = hr_img.crop((x, y, x + patch_size, y + patch_size))
            lr_patch = lr_img_up.crop((x, y, x + patch_size, y + patch_size))
            
            filename = f"{patch_idx:06d}.png"
            hr_patch.save(os.path.join(save_dir, "HR", filename))
            lr_patch.save(os.path.join(save_dir, "LR", filename))
            
            patch_idx += 1
            
    print(f"Finished! Total {patch_idx} patches saved in {save_dir}")

if __name__ == "__main__":
    # Train data
    prepare_dataset_lr(
        hr_dir="./DIV2K/DIV2K_train_HR",
        lr_dir="./DIV2K/DIV2K_train_LR_bicubic/X4",
        save_dir="./data_lr/train",
        patches_per_img=20 # 800 * 20 = 16,000 patches
    )
    # Valid data
    prepare_dataset_lr(
        hr_dir="./DIV2K/DIV2K_valid_HR",
        lr_dir="./DIV2K/DIV2K_valid_LR_bicubic/X4",
        save_dir="./data_lr/valid",
        patches_per_img=5 # 100 * 5 = 500 patches
    )
