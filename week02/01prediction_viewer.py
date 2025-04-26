import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button
import sys
from model import FaceDetector
from torch.utils.data import ConcatDataset 
from yoloy_dataset import YoloDataset
from utils import yolo_label_to_cxcywh
from nonmax_suppression import non_max_supp
import os


class ImagePredictionViewer:
    def __init__(self, images, all_preds, all_labels=None, preds_type='cx,cy,w,h', labels_type='cx,cy,w,h'):
        self.images = images  # List of images
        self.all_preds = all_preds  # List of lists of predictions
        self.all_labels = all_labels if all_labels is not None else [[] for _ in range(len(images))]
        self.preds_type = preds_type
        self.labels_type = labels_type
        self.current_img_index = 0
        self.num_images = len(images)
        
        # Create figure and connect key events
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Display the first image with all its predictions
        self.update_plot()
        
        plt.tight_layout()
        plt.show()
    
    def update_plot(self):
        self.ax.clear()
        
        # Get current image and convert if needed
        img = self.images[self.current_img_index]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)  # Convert from (3 x W x H) to (W x H x 3)
        
        self.ax.imshow(img)
        img_h, img_w = img.shape[:2]
        
        # Show current image index / total images
        self.ax.set_title(f"Image {self.current_img_index + 1}/{self.num_images}")
        
        # Plot all predictions for this image
        current_preds = self.all_preds[self.current_img_index]
        for pred in current_preds:
            if self.preds_type == 'cx,cy,w,h':
                self.plot_box(pred, img_w, img_h, 'red', 'ro', True)
        
        # Plot all ground truth labels for this image
        current_labels = self.all_labels[self.current_img_index]
        for label in current_labels:
            if self.labels_type == 'cx,cy,w,h':
                self.plot_box(label, img_w, img_h, 'green', 'g1', False)
        
        self.ax.axis('off')
        self.fig.canvas.draw_idle()
    
    def plot_box(self, box, img_w, img_h, color, marker, show_confidence):
        confidance, center_x, center_y, width, height = box
        center_x, center_y = int(center_x * img_w), int(center_y * img_h)
        width, height = int(width * img_w), int(height * img_h)
        
        xmin = center_x - width//2
        xmax = center_x + width//2
        ymin = center_y - height//2
        ymax = center_y + height//2
        
        # Plot center point
        self.ax.plot(center_x, center_y, marker)
        
        # Show confidence score
        if show_confidence:
            conf_value = confidance.item() if hasattr(confidance, 'item') else confidance
            self.ax.text(center_x, center_y, f"{conf_value:.3f}", 
                     bbox=dict(facecolor='white', alpha=0.35, boxstyle='round'),
                     fontsize=8)
        
        # Draw bounding box
        self.ax.hlines(ymin, xmin=xmin, xmax=xmax, colors=color)
        self.ax.hlines(ymax, xmin=xmin, xmax=xmax, colors=color)
        self.ax.vlines(xmin, ymin=ymin, ymax=ymax, colors=color)
        self.ax.vlines(xmax, ymin=ymin, ymax=ymax, colors=color)
    
    def on_key_press(self, event):
        if event.key == 'right' and self.current_img_index < self.num_images - 1:
            self.current_img_index += 1
            self.update_plot()
        elif event.key == 'left' and self.current_img_index > 0:
            self.current_img_index -= 1
            self.update_plot()
        elif event.key == 'escape':
            plt.close(self.fig)

# Example usage (this part would be in your script)
if __name__ == "__main__":
    # Replace this with your actual data loading code
    # For example:
    # images = [torch.load(f'image_{i}.pt') for i in range(num_images)]
    # all_preds = [torch.load(f'preds_{i}.pt') for i in range(num_images)]
    # all_labels = [torch.load(f'labels_{i}.pt') for i in range(num_images)]
    
    PATH_TO_MODEL = '/Users/kuba/projects/swvl/week02/best_dev_loss.pt'
    PATH_TO_TEST_DIR = '/Users/kuba/Documents/data/raw/single-face-tensors/test'

    #set up model
    model = FaceDetector()
    model.load_state_dict(torch.load(PATH_TO_MODEL, weights_only=True, map_location=torch.device('cpu')))

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
    
    images = []
    all_labels = []

    # Get 5 items individually
    for i in range(25):
        if i < len(combined):
            image, labels = combined[i]
            images.append(image)
            all_labels.append(labels)
    images = torch.stack(images)
    all_preds = model(images)
    all_preds = all_preds.view(-1, 7, 7, 10)
    all_preds = non_max_supp(all_preds, return_type='xywh')
    new_all_lables = []
    for lable in all_labels:
        new_all_lables.append(yolo_label_to_cxcywh(lable, (224,224)))

    print(torch.equal(images[0], images[1]))

    # Create viewer
    viewer = ImagePredictionViewer(images, all_preds, new_all_lables)