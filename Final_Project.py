import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("DistributedUNet").getOrCreate()


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = CBR(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, 25, kernel_size=1)  # 25个类别

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        center = self.center(self.pool(enc4))
        dec4 = self.dec4(torch.cat([self.up4(center), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        final = self.final(dec1)
        return final


def process_data(image_mask_pair):
    image_path, mask_path = image_mask_pair
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # 灰度掩码
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image = transform(image)
    mask = transform(mask)
    return image, mask


def load_data_rdd(images_dir, masks_dir):
    images_list = os.listdir(images_dir)
    masks_list = [image.replace(".jpg", ".png") for image in images_list]
    image_mask_paths = [(os.path.join(images_dir, img), os.path.join(masks_dir, mask)) for img, mask in zip(images_list, masks_list)]
    data_rdd = spark.sparkContext.parallelize(image_mask_paths, numSlices=2)
    return data_rdd


def train_on_partition(iterator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for _ in range(5):  
        running_loss = 0.0
        for data in iterator:
            image, mask = process_data(data)
            image, mask = image.to(device), mask.to(device).long().squeeze(0)

            optimizer.zero_grad()
            outputs = model(image.unsqueeze(0)) 
            loss = criterion(outputs, mask.unsqueeze(0))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        yield running_loss


train_images_dir = "C:/Users/yuhon/Big_data/Final_project/Data/harvey_damage_satelite/train_images"
train_masks_dir = "C:/Users/yuhon/Big_data/Final_project/Data/harvey_damage_satelite/train_masks"
train_data_rdd = load_data_rdd(train_images_dir, train_masks_dir)


losses = train_data_rdd.mapPartitions(train_on_partition).collect()
print(f"Total training loss across partitions: {sum(losses)}")

