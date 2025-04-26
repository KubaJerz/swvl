from model import FaceDetector
from torch.utils.data import DataLoader, Subset
import os
import random
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from loss_multiface import Yolo_Loss
from full_face_dataset import FullFaceDatasetOptimized
import json


HYPERPARAM_OPTIONS = {
    "lr": [1e-3, 1e-4, 1e-5, 1e-6],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65],
    "weight_decay": [0.01, 0.001, 0.0005]
}

# configuration
NUM_TRYS = 16
EPOCHS = 200
SAVE_DIR = './experiments'

# data paths
TRAIN_CSV_PATH = '/home/kuba/Documents/data/raw/face-detection-dataset/train.csv'
TRAIN_IMG_DIR = '/home/kuba/Documents/data/raw/face-detection-dataset/images/train'
TRAIN_LABELS_DIR = '/home/kuba/Documents/data/raw/face-detection-dataset/labels/train' 

# create dataset and split into train/dev
full_dataset = FullFaceDatasetOptimized(TRAIN_CSV_PATH, TRAIN_IMG_DIR, TRAIN_LABELS_DIR)
train_indices = list(range(10000))
dev_indices = list(range(10000, len(full_dataset)))
# train_indices = list(range(10))
# dev_indices = list(range(10, 15))

# create subsets
train_dataset = Subset(full_dataset, train_indices)
dev_dataset = Subset(full_dataset, dev_indices)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=64)
dev_loader = DataLoader(dev_dataset, batch_size=128)


def sample_hyperparams():
    """Sample hyperparameters randomly from the predefined options."""
    hyper_params = {}
    for param, options in HYPERPARAM_OPTIONS.items():
        hyper_params[param] = random.choice(options)
    return hyper_params


def train(hyper_params):
    """Train the model with the given hyperparameters."""
    dropout = hyper_params['dropout']
    model = FaceDetector(dropout=dropout)

    # freeze backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False

    # setup optimizer
    optimizer = torch.optim.SGD(
        model.regression_head.parameters(), 
        lr=hyper_params['lr'], 
        momentum=0.9, 
        weight_decay=hyper_params['weight_decay']
    )

    # setup loss function and scheduler
    criterion = Yolo_Loss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=10)
    
    # get scheduler info for logging
    lr_scheduler_name = 'None'
    lr_scheduler_info = "N/A"
    if scheduler is not None:
        lr_scheduler_name = scheduler.__class__.__name__
        lr_scheduler_info = [f"{key}:{val}" for key, val in scheduler.state_dict().items() 
                            if key != 'state' and key != '_step_count']

    NAME = f"full_lr{hyper_params['lr']}_dropout{dropout}_wd{hyper_params['weight_decay']}"
    
    # create  dir
    os.makedirs(SAVE_DIR+'/'+NAME, exist_ok=True)

    # log  details
    notes = f"""
    optimizer type: {optimizer.__class__.__name__} lr: {optimizer.param_groups[0]['lr']} 
    Other parameters: {[f"{key}: {val}" for key, val in optimizer.param_groups[0].items() if key != 'params' and key != 'lr']}   
    \nEPOCHS: {EPOCHS}
    \nBatchsize: train:{train_loader.batch_size} dev:{dev_loader.batch_size}
    \nDropout: {dropout}
    \nscheduler type: {lr_scheduler_name} 
    parameters: {lr_scheduler_info}
    \nDataset: FULL
    """

    with open(os.path.join(SAVE_DIR, NAME, "notes.txt"), "w") as file:
        file.write(notes)

    lossi = []
    dev_lossi = []
    best_dev_loss = float('inf')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        loss_total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        
        avg_train_loss = loss_total / len(train_loader)
        lossi.append(avg_train_loss)

        model.eval()
        dev_loss_total = 0
        with torch.no_grad():
            for X_dev, y_dev in dev_loader:
                X_dev, y_dev = X_dev.to(device), y_dev.to(device)
                dev_logits = model(X_dev)
                dev_loss = criterion(dev_logits, y_dev)
                dev_loss_total += dev_loss.item()
        
        avg_dev_loss = dev_loss_total / len(dev_loader)
        dev_lossi.append(avg_dev_loss)
        
        scheduler.step()

        # Save best model
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            print(f'Epoch: {epoch+1} Dev Loss: {best_dev_loss:.6f}')
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, NAME, "best_dev_loss.pt"))

        # plot loss curve
        if (epoch + 1) % 2 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(lossi, label='train Loss')
            plt.plot(dev_lossi, label='dev Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.savefig(os.path.join(SAVE_DIR, NAME, "loss.png"))
            plt.close()

        # save metrics every 5
        if (epoch + 1) % 5 == 0:
            with open(os.path.join(SAVE_DIR, NAME, "metrics.json"), "w") as file:
                json.dump({"lossi": lossi, "devlossi": dev_lossi}, file)


if __name__ == "__main__":
    for try_num in range(NUM_TRYS):
        print(f"Starting hyperparameter trial {try_num+1}/{NUM_TRYS}")
        hyper_params = sample_hyperparams()
        print(f"Hyperparameters: {hyper_params}")
        train(hyper_params)