import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button
import sys

class PredictionViewer:
    def __init__(self, img, preds, preds_type='cx,cy,w,h', labels=[], labels_type='cx,cy,w,h'):
        self.img = img.permute(1, 2, 0) if isinstance(img, torch.Tensor) else img
        self.preds = preds
        self.preds_type = preds_type
        self.labels = labels
        self.labels_type = labels_type
        self.current_index = 0
        self.max_index = len(preds) - 1
        
        # Create figure and connect key events
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Display the first prediction
        self.update_plot()
        
        plt.tight_layout()
        plt.show()
    
    def update_plot(self):
        self.ax.clear()
        self.ax.imshow(self.img)
        img_h, img_w = self.img.shape[:2]
        
        # Show current index / total predictions
        self.ax.set_title(f"Prediction {self.current_index + 1}/{self.max_index + 1}")
        
        # Plot current prediction
        if self.preds_type == 'cx,cy,w,h':
            self.plot_box(self.preds[self.current_index], img_w, img_h, 'red', 'ro', True)
        
        # Plot ground truth label if available
        if len(self.labels) > 0 and self.labels_type == 'cx,cy,w,h':
            for label in self.labels:
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
        if event.key == 'right' and self.current_index < self.max_index:
            self.current_index += 1
            self.update_plot()
        elif event.key == 'left' and self.current_index > 0:
            self.current_index -= 1
            self.update_plot()
        elif event.key == 'escape':
            plt.close(self.fig)

# Example usage (this part would be in your script)
if __name__ == "__main__":
    # Replace this with your actual data loading code
    # For example:
    # img = torch.load('your_image.pt')
    # preds = torch.load('your_predictions.pt')
    # labels = torch.load('your_labels.pt')
    
    # Dummy data for example
    img = torch.ones(3, 224, 224)
    preds = [
        [0.95, 0.2, 0.3, 0.1, 0.2],
        [0.85, 0.5, 0.6, 0.2, 0.3],
        [0.75, 0.7, 0.7, 0.15, 0.25]
    ]
    labels = [[1.0, 0.2, 0.3, 0.1, 0.2]]
    
    # Create viewer
    viewer = PredictionViewer(img, preds, labels=labels)