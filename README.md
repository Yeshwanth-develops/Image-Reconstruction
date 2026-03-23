# 🧠 Image Reconstruction from Occluded Scenes 
  ### (U-Net + CelebA)

A deep learning project that reconstructs missing regions in images using U-Net architecture and hybrid loss functions. The model learns contextual information to generate realistic image completions from partially occluded inputs.

## 🔍 Reconstruction Examples
Input: Image with random occlusion (black mask)

Output: Reconstructed image using trained model

<img width="398" height="135" alt="{4436EA2A-1836-4DF1-B245-F7374598E484}" src="https://github.com/user-attachments/assets/80800ebd-e8e9-489f-b6dd-475e70f5b80d" />

<img width="397" height="135" alt="{44EC6539-CC60-4748-B413-484AF8F3E20D}" src="https://github.com/user-attachments/assets/b29e5ee7-429f-4d1f-b633-55acd07ea138" />

<img width="397" height="135" alt="{39204590-A65B-411B-A0FD-3603F27CB338}" src="https://github.com/user-attachments/assets/bacc4ccc-45fc-43d3-ba1d-1e9266b6918b" />


## 📌 Project Overview

This project focuses on image inpainting, where a model learns to fill missing or corrupted parts of an image using surrounding context.

Key Highlights:

Uses U-Net (Encoder-Decoder with skip connections)

Trained on CelebA dataset (128×128 resolution)

Simulates real-world occlusions using random masks

Combines L1 Loss + SSIM Loss for better reconstruction quality

## 🛠️ Tech Stack

Language: Python

Framework: PyTorch

Libraries: torchvision, torchmetrics, OpenCV, NumPy, Matplotlib

Dataset: CelebA

## 🧠 Model Architecture

Encoder: Extracts hierarchical features

Bottleneck: Captures latent representation

Decoder: Reconstructs image using skip connections

Skip Connections: Preserve spatial details

## ⚙️ Training Details

Parameter	Value

Image Size	128 × 128

Batch Size	32

Epochs	10–20

Optimizer	Adam

Learning Rate	0.0003

Loss Function	L1 + 0.3 × SSIM

## 🎭 Occlusion Strategy

Random square masks are applied to simulate missing regions:

Mask Size: 24–64 pixels

Position: Random per image

Purpose: Improve generalization and robustness

## 📊 Evaluation Metrics

L1 Loss → Pixel-wise difference

SSIM (Structural Similarity Index) → Perceptual similarity

## 📈 Results

Successfully reconstructs occluded facial regions

Preserves:

Face structure

Skin tone consistency

Background continuity

Observations:

Larger masks → more challenging reconstruction

More epochs → sharper outputs

Slight blurring due to pixel-based losses

## 📂 Project Structure
```text
image-reconstruction-unet/
│
├── dataset/
├── models/
│   └── unet.py
├── train.py
├── evaluate.py
├── utils.py
└── README.md
```

## ▶️ How to Run

### 1️⃣ Install dependencies

pip install torch torchvision matplotlib numpy opencv-python tqdm torchmetrics kagglehub

### 2️⃣ Download dataset

import kagglehub
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

### 3️⃣ Train the model

python train.py

### 4️⃣ Evaluate

python evaluate.py

## 💡 Key Learnings

U-Net significantly improves reconstruction vs basic autoencoders

SSIM helps preserve structural similarity

Dataset quality (CelebA vs CIFAR) greatly impacts output

Larger image resolution improves visual realism

## 🚀 Future Improvements

🔥 Add Perceptual Loss (VGG-based)

🎨 Use GAN-based inpainting (Context Encoder)

🧩 Implement irregular masks instead of square masks

👁️ Add attention mechanism

⚡ Optimize training with mixed precision

## 🎯 Applications

Face restoration

Medical image reconstruction

Photo editing / object removal

Surveillance image recovery

Autonomous driving vision systems
