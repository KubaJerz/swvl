import torch
from model import FaceDetector
from torch.utils.data import ConcatDataset, DataLoader
import os
from yoloy_dataset import YoloDataset
from loss import Yolo_Loss
from nonmax_suppression import non_max_supp
from utils import yolo_labels_to_xyxy
from mAP import mAP
from tqdm import tqdm


PATH_TO_MODEL = '/home/kuba/projects/swvl/week02/pp00.pt'
PATH_TO_TEST_DIR = '/home/kuba/Documents/data/raw/single-face-tensors/test'

#set up model
model = FaceDetector()
model.load_state_dict(torch.load(PATH_TO_MODEL, weights_only=True))

model.eval()

#setup test data

all_train_datasets = []

for file in sorted(os.listdir(PATH_TO_TEST_DIR)):
    try:
        dataset = YoloDataset((PATH_TO_TEST_DIR+"/"+file))
        all_train_datasets.append(dataset)
    except Exception as e:
        print(f"Error loading {file}: {str(e)}")
        continue

combined = ConcatDataset(all_train_datasets)

test_loader = DataLoader(combined, batch_size=512)
criterion = Yolo_Loss()

all_logits = []
all_labels = []

device = 'cuda:1'
model = model.to(device)

loss_total = 0
lossi = []
for X_batch, y_batch in tqdm(test_loader):
    with torch.no_grad():
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        test_logits = model(X_batch)
        all_logits.append(test_logits.cpu())
        all_labels.append(y_batch.cpu())
        loss = criterion(test_logits, y_batch)
        loss_total += loss.item()
        lossi.append(loss.item())

loss_total = loss_total / len(test_loader)

all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)

print(all_logits.shape)
print(all_labels.shape)

predictions = all_logits.view(-1, 7, 7, 10)
truths = all_labels.view(-1, 7, 7, 5)

supp_predictions = non_max_supp(predictions, return_type='xyxy')
truths_xyxy = yolo_labels_to_xyxy(truths)

map = mAP(supp_predictions, truths_xyxy)


print(f"map is: {map}")
print(f"avg loss is: {loss_total}")
