import os
import sys
import glob
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

import importlib.util

# Load VDSR model directly from path to avoid clashing with Frequency/model.py
spec = importlib.util.spec_from_file_location("vdsr_model", "../VDSR/model.py")
vdsr_module = importlib.util.module_from_spec(spec)
sys.modules["vdsr_model"] = vdsr_module
spec.loader.exec_module(vdsr_module)
VDSR = vdsr_module.VDSR

def prepare_dataset(hr_dir, save_dir, model_weight_path, patch_size=256, scale_factor=4, patches_per_img=5):
    # Setup directories
    lr_dir = os.path.join(save_dir, "LR")
    sr_dir = os.path.join(save_dir, "SR")
    hr_save_dir = os.path.join(save_dir, "HR")
    
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(hr_save_dir, exist_ok=True)

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = VDSR(in_channels=3).to(device)
    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print(f"Loaded VDSR model weights from {model_weight_path}")
    else:
        print(f"Warning: Model weights not found at {model_weight_path}. Using random weights!")
    model.eval()

    # Image transforms
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    image_paths = glob.glob(os.path.join(hr_dir, "*.png"))
    image_paths = sorted(image_paths)
    
    print(f"Found {len(image_paths)} HR images. Extracting {patches_per_img} patches per image...")
    
    patch_idx = 0
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Processing Images"):
            image = Image.open(img_path).convert("RGB")
            
            # Extract random crops
            for _ in range(patches_per_img):
                # Random Crop
                crop_transform = transforms.RandomCrop(patch_size)
                hr_patch = crop_transform(image)
                
                # Downsample and Upsample (Bicubic) to create LR input
                lr_patch = hr_patch.resize(
                    (patch_size // scale_factor, patch_size // scale_factor), 
                    Image.BICUBIC
                )
                lr_patch_up = lr_patch.resize(
                    (patch_size, patch_size), 
                    Image.BICUBIC
                )
                
                # Inference to get SR
                lr_tensor = to_tensor(lr_patch_up).unsqueeze(0).to(device) # [1, 3, H, W]
                residual, outputs = model(lr_tensor)
                
                # Clamp outputs to [0, 1] to be valid images
                outputs = torch.clamp(outputs, 0.0, 1.0)
                sr_patch = to_pil(outputs.squeeze(0).cpu())
                
                # Save patches
                filename = f"{patch_idx:05d}.png"
                lr_patch_up.save(os.path.join(lr_dir, filename))
                sr_patch.save(os.path.join(sr_dir, filename))
                hr_patch.save(os.path.join(hr_save_dir, filename))
                
                patch_idx += 1
                
    print(f"Dataset preparation complete. Total {patch_idx} patches saved in {save_dir}")

if __name__ == "__main__":
    train_hr_dir = "../VDSR/DIV2K/DIV2K_train_HR"
    save_directory = "./data/train"
    weight_path = "../VDSR/weights/vdsr_epoch_100.pth"
    
    # Generate 1000 patches (200 images * 5) for quick training, or process all.
    # We'll just process everything. It's fast.
    prepare_dataset(train_hr_dir, save_directory, weight_path, patch_size=256, patches_per_img=5)
