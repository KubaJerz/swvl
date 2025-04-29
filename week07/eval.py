import torch
from model import FaceDetector
from torch.utils.data import ConcatDataset, DataLoader
import os
from full_face_dataset import FullFaceDataset
from loss_multiface import Yolo_Loss
from nonmax_suppression import non_max_supp
from utils import yolo_labels_to_xyxy
from mAP import mAP
from tqdm import tqdm
from torchvision import transforms

PATH_TO_MODEL = '/home/kuba/projects/swvl/week07/experiments/02/best_dev_loss0476_dsc.pth'

#----------------
TEST_CSV_PATH = '/home/kuba/Documents/data/raw/face-detection-dataset/test.csv'
PATH_IMG_TEST_DIR = '/home/kuba/Documents/data/raw/face-detection-dataset/images/val'
PATH_LABELS_TEST_DIR = '/home/kuba/Documents/data/raw/face-detection-dataset/labels/val'

#set up model
model = FaceDetector()
model.load_state_dict(torch.load(PATH_TO_MODEL, weights_only=True))

model.eval()

#setup test data


transform_img_net =  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
full_dataset = FullFaceDataset(csv_file=TEST_CSV_PATH, img_dir=PATH_IMG_TEST_DIR, label_dir=PATH_LABELS_TEST_DIR, transforms=transform_img_net, target_size=(448, 448))

test_loader = DataLoader(full_dataset, batch_size=256)
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

print(f"eval model at: {os.path.join(*(PATH_TO_MODEL).split(sep='/')[-2:])}")
print(f"map is: {map}")
print(f"avg loss is: {loss_total}")
