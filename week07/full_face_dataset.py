import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms

class FullFaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C = 0, target_size=(448, 448),transforms=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.target_size = target_size
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                confidance, x, y, width, height = [float(x) for x in label.replace("\n", "").split()]
                boxes.append([confidance, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        boxes = torch.tensor(boxes)
        basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.target_size, antialias=True),  # Resize all images to same dimensions
        ])
        image = basic_transform(image)
        if image.size()[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.size(0) == 4:
            #keep only the first 3 channels (RGB)
            image = image[:3, :, :]

        if self.transforms:
            image = self.transforms(image)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            _, x, y, w, h = box.tolist()
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            w_cell, h_cell = w * self.S, h * self.S

            if label_matrix[i, j, 0] == 0:
                    label_matrix[i, j, 0] = 1
                    label_matrix[i, j, 1:5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
            # elif label_matrix[i, j, 0] == 1:
            #     print(f"Warning: Loading Img: {img_path}\nTryed to add label to cell:({i},{j}) but another label already present")
        
        return image, label_matrix
    
class FullFaceDatasetOptimized(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C = 0, target_size=(448, 448),transforms=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.target_size = target_size
        self.S = S
        self.B = B
        self.C = C
        self.cache = {} 

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                confidance, x, y, width, height = [float(x) for x in label.replace("\n", "").split()]
                boxes.append([confidance, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        boxes = torch.tensor(boxes)
        basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.target_size, antialias=True),  # Resize all images to same dimensions
        ])
        image = basic_transform(image)
        if image.size()[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.size(0) == 4:
            #keep only the first 3 channels (RGB)
            image = image[:3, :, :]

        if self.transforms:
            image = self.transforms(image)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            _, x, y, w, h = box.tolist()
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            w_cell, h_cell = w * self.S, h * self.S

            if label_matrix[i, j, 0] == 0:
                    label_matrix[i, j, 0] = 1
                    label_matrix[i, j, 1:5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
            # elif label_matrix[i, j, 0] == 1:
            #     print(f"Warning: Loading Img: {img_path}\nTryed to add label to cell:({i},{j}) but another label already present")
        
        self.cache[index] = (image, label_matrix)
        
        return image, label_matrix