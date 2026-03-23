import sys
!{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install matplotlib numpy opencv-python tqdm scikit-image torchmetrics kagglehub

import kagglehub

path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

print("Dataset path:", path)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.datasets import ImageFolder

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3,64)
        self.enc2 = DoubleConv(64,128)
        self.enc3 = DoubleConv(128,256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256,512)

        self.up3 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.dec3 = DoubleConv(512,256)

        self.up2 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec2 = DoubleConv(256,128)

        self.up1 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec1 = DoubleConv(128,64)

        self.final = nn.Conv2d(64,3,1)
        self.act = nn.Sigmoid()

    def forward(self,x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3,e3],dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2,e2],dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1,e1],dim=1)
        d1 = self.dec1(d1)

        return self.act(self.final(d1))

def add_occlusion_batch(images):

    images = images.clone()

    b,c,h,w = images.shape

    for i in range(b):

        mask_size = random.randint(24,48)

        x = random.randint(0,h-mask_size)
        y = random.randint(0,w-mask_size)

        images[i,:,x:x+mask_size,y:y+mask_size] = 0

    return images

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

dataset = ImageFolder(
    root = path + "/img_align_celeba",
    transform = transform
)

trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 2
)

testloader = torch.utils.data.DataLoader(
    dataset,
    batch_size = 32,
    shuffle = False,
    num_workers = 2
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)

l1_loss_fn = nn.L1Loss()

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0003)

EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for images,_ in tqdm(trainloader):

        images = images.to(device)

        occluded = add_occlusion_batch(images)

        outputs = model(occluded)

        l1 = l1_loss_fn(outputs,images)
        ssim = 1 - ssim_metric(outputs,images)

        loss = l1 + 0.3*ssim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss {train_loss/len(trainloader):.4f}")

    torch.save(model.state_dict(), "celeba_unet_inpainting.pth")

import matplotlib.pyplot as plt

def show_images(original,occluded,reconstructed):

    fig,axs = plt.subplots(1,3,figsize=(10,4))

    axs[0].imshow(original.permute(1,2,0))
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(occluded.permute(1,2,0))
    axs[1].set_title("Occluded")
    axs[1].axis("off")

    axs[2].imshow(reconstructed.permute(1,2,0))
    axs[2].set_title("Reconstructed")
    axs[2].axis("off")

    plt.show()


model.eval()

dataiter = iter(testloader)
images,_ = next(dataiter)

for i in range(3):

    original = images[i].to(device)

    occluded = add_occlusion_batch(original.unsqueeze(0)).squeeze(0)

    with torch.no_grad():
        reconstructed = model(occluded.unsqueeze(0)).squeeze(0)

    show_images(
        original.cpu(),
        occluded.cpu(),
        reconstructed.cpu()
    )