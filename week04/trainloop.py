from model import FaceDetector
from torch.utils.data import ConcatDataset, DataLoader, Subset
import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from loss_multiface import Yolo_Loss
from full_face_dataset import FullFaceDataset, FullFaceDatasetOptimized
import numpy as np
import json

TRAIN_CSV_PATH = '/home/kuba/Documents/data/raw/face-detection-dataset/train.csv'
TRAIN_IMG_DIR = '/home/kuba/Documents/data/raw/face-detection-dataset/images/train'
TRAIN_LABELS_DIR = '/home/kuba/Documents/data/raw/face-detection-dataset/labels/train' 
full_dataset = FullFaceDatasetOptimized(TRAIN_CSV_PATH, TRAIN_IMG_DIR, TRAIN_LABELS_DIR)
train_indices = list(range(10000))
dev_indices = list(range(10000, len(full_dataset)))

#make subsets
train_dataset = Subset(full_dataset, train_indices)
dev_dataset = Subset(full_dataset, dev_indices)

#make DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64)
dev_loader = DataLoader(dev_dataset, batch_size=128)


# _____________________________________
dropout = 0.5
model = FaceDetector(dropout=dropout)

for param in model.backbone.parameters():
    param.requires_grad = False

# optimizer = torch.optim.AdamW(model.regression_head.parameters(), lr=0.00003, amsgrad=True, weight_decay=0.0005)
optimizer = torch.optim.SGD(model.regression_head.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

criterion = Yolo_Loss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=10)
lr_scheduler_name = 'None'
lr_scheduler_info = "N/A"
if scheduler != None:
    lr_scheduler_name = scheduler.__class__.__name__
    lr_scheduler_info = [f"{key}:{val}" for key, val in scheduler.state_dict().items() if key != 'state' and key != '_step_count']


EPOCHS = 250
NAME = "full00"
SAVE_DIR = './experiments'


notes = f"""
optimizer type: {optimizer.__class__.__name__} lr: {optimizer.param_groups[0]['lr']} 
Other paramiters: {[f"{key}: {val}" for key, val in optimizer.param_groups[0].items() if key != 'params' and key != 'lr']}   
\nEPOCHS: {EPOCHS}
\nBatchsize: train:{train_loader.batch_size} dev:{dev_loader.batch_size}
\nDropout: {dropout}
\nscheduler type: {lr_scheduler_name} 
paramiters: {lr_scheduler_info}
\nDataset: FULL
"""

# ______________________________________

os.mkdir(SAVE_DIR+'/'+NAME)

with open(SAVE_DIR+'/'+NAME+"/"+"notes.txt", "w") as file:
    file.write(notes)



lossi = []
dev_lossi = []

best_dev_loss = float('inf')

device = 'cuda:0'
model = model.to(device)

for epoch in tqdm(range(EPOCHS)):
    loss_total = 0
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    lossi.append(loss_total / len(train_loader))



    dev_loss_total = 0
    model.eval()
    with torch.no_grad():
        for X_dev, y_dev in dev_loader:
            X_dev, y_dev = X_dev.to(device), y_dev.to(device)
            dev_logits = model(X_dev)
            dev_loss = criterion(dev_logits, y_dev)
            dev_loss_total += dev_loss.item()
    dev_lossi.append(dev_loss_total/ len(dev_loader))
    scheduler.step()

    if (dev_loss_total/ len(dev_loader)) < best_dev_loss:
        best_dev_loss = (dev_loss_total/ len(dev_loader))
        print(f'Epoch: {epoch+1} Dev Loss:{best_dev_loss}')
        torch.save(model.state_dict(), SAVE_DIR+'/'+NAME+"/"+"best_dev_loss.pt")

    if (epoch + 1) % 2 == 0:

        plt.plot( lossi[1:], label='lossi')
        plt.plot( dev_lossi[1:], label='devlossi')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(SAVE_DIR+'/'+NAME+"/"+"loss.png")
        plt.close()

    if (epoch + 1) % 5 == 0:
        with open(SAVE_DIR+'/'+NAME+"/"+"metrics.json", "w") as file:
            json.dump({"lossi":lossi, "devlossi":dev_lossi}, file)
    



