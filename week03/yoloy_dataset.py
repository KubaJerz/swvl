
import torch
from torch.utils.data import Dataset

class YoloDataset(Dataset):
    def __init__(self, path, S=7, B=2, C = 0):
        super().__init__()
        self.X , self.y = torch.load(path)
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        image = self.X
        boxes = self.y
        labels = torch.zeros((self.S, self.S, self.C + 5))
        
        # single box 
        if boxes.dim() == 1 and len(boxes) == 4:
            x, y, w, h = boxes.tolist()
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            w_cell, h_cell = w * self.S, h * self.S
            
            if labels[i, j, 0] == 0:
                labels[i, j, 0] = 1
                labels[i, j, 1:5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
        # multiple boxes
        elif boxes.dim() > 1:
            for unscaled_box in boxes:
                x, y, w, h = unscaled_box.tolist()
                i, j = int(self.S * y), int(self.S * x)
                x_cell, y_cell = self.S * x - j, self.S * y - i
                w_cell, h_cell = w * self.S, h * self.S
                
                if labels[i, j, 0] == 0:
                    labels[i, j, 0] = 1
                    labels[i, j, 1:5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
        
        return image, labels