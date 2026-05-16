#!/bin/bash

# Create DIV2K directory
mkdir -p DIV2K
cd DIV2K

echo "Starting DIV2K dataset download..."

# HR Images
echo "Downloading HR images..."
wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

# LR Images (Bicubic x2, x3, x4)
echo "Downloading LR bicubic images..."
wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip
wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip
wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip

echo "Download complete. Starting extraction..."

# Extract all zip files
for f in *.zip; do
    echo "Extracting $f..."
    unzip -q "$f"
done

echo "Cleaning up zip files..."
# Optional: keep zips or remove them. I'll keep them for safety for now unless asked.
# rm *.zip

echo "DIV2K dataset preparation complete!"
