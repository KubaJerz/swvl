import torch
import torch.nn as nn
from torchvision.ops import box_iou, box_convert

def iou(preds, targets):
    # Convert (x, y, width, height) to (x1, y1, x2, y2)
    pred_x1 = preds[..., 0] - preds[..., 2] / 2
    pred_y1 = preds[..., 1] - preds[..., 3] / 2
    pred_x2 = preds[..., 0] + preds[..., 2] / 2
    pred_y2 = preds[..., 1] + preds[..., 3] / 2
        
    target_x1 = targets[..., 0] - targets[..., 2] / 2
    target_y1 = targets[..., 1] - targets[..., 3] / 2
    target_x2 = targets[..., 0] + targets[..., 2] / 2
    target_y2 = targets[..., 1] + targets[..., 3] / 2
    
    # Calculate intersection area
    x1 = torch.max(pred_x1, target_x1)
    y1 = torch.max(pred_y1, target_y1)
    x2 = torch.min(pred_x2, target_x2)
    y2 = torch.min(pred_y2, target_y2)
    
    intersection = torch.clamp((x2 - x1), min=0) * torch.clamp((y2 - y1), min=0)
    
    # Calculate union area
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union = pred_area + target_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-7)
    return iou

# This is YOLO loss but we have no class probs just one class
class Yolo_Loss(nn.Module):
    def __init__(self, S=7, B=2):
        super().__init__()
        self.mse = nn.MSELoss(reduce="sum")
        self.S = S
        self.B = B

        self.lambda_coord = 5
        self.lambda_no_obj = 0.5


    def forward(self, prediction, target):
        prediction = prediction.view(-1, self.S, self.S, self.B*5)
        iou0 = iou(prediction[..., 1:5], target[..., 1:5]) #prediction is (S*S, B*5) => (S*S, B*(confidence, x,y,wh))
        iou1 = iou(prediction[..., 6:10], target[..., 1:5]) #prediction is (S*S, B*5) => (S*S, B*(confidence, x,y,wh))
        ious = torch.cat([iou0.unsqueeze(0), iou1.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        bestbox = bestbox.unsqueeze(dim=-1)

        exits_box = target[..., 0].unsqueeze(3)

        # For box coord
        #best box tells use which box has teh higher iou then we use that as teh prediction
        box_preds = exits_box * ((bestbox * prediction[..., 6:10]) + ((1- bestbox) * prediction[..., 1:5]))
        # Take the square root of width and height to handle size boxe differences
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(torch.abs(box_preds[..., 2:4] + 1e-7))
    

        #since we are prediction only one class the target vector will be (S,S,5)
        box_targets = exits_box * target[..., 1:5]
        #sqrt of the target witdh and height too
        box_targets[..., 2:4] = torch.sign(box_targets[..., 2:4]) * torch.sqrt(torch.abs(box_targets[..., 2:4]) + 1e-7)
        
        # we flatten so that is (N*S*S, 4) and we jsut coment the values
        box_loss = self.mse(box_preds.flatten(end_dim=-2), box_targets.flatten(end_dim=-2))

        #for the next part of teh loss we need to do confidence of object fior the ones that do have a true object
        pred_confidence = (bestbox * prediction[..., 5:6]) + ((1- bestbox) * prediction[..., 0:1])
        target_confidence = target[..., 0:1]

        # OBJECT LOSS
        # This is where there is a box and model shouwl have predicated it
        obj_loss = self.mse((exits_box * pred_confidence).flatten(end_dim=-2), (exits_box * target_confidence).flatten(end_dim=-2))
        # obj_loss = self.mse(exits_box * pred_confidence, (1 - exits_box) * target_confidence)


        # NO OBJECT LOSS
        # This is for when the model predicts an obj but their is non
        no_obj00_loss = self.mse(((1 - exits_box) * prediction[..., 0:1]).flatten(end_dim=-2), ((1 - exits_box) * target_confidence).flatten(end_dim=-2))
        no_obj01_loss = self.mse(((1 - exits_box) * prediction[..., 5:6]).flatten(end_dim=-2), ((1 - exits_box) * target_confidence).flatten(end_dim=-2))

        loss = (self.lambda_coord * box_loss + obj_loss + self.lambda_no_obj * (no_obj00_loss + no_obj01_loss))

        return loss
