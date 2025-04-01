import numpy as np
import torch
import torchvision.ops as ops

def mAP(preds, labels, iou_start=0.5, iou_stop=0.95, iou_step=0.05):
    # expected shape for preds is (BatchSize x num_preds x 5)
    # expected shape for labels is (BatchSize x num_true_labels x 5)
    
    mAPs = [] #for each iou  threshold
    
    for iou_thresh in np.arange(start=iou_start, stop=iou_stop + iou_step/2, step=iou_step):
        batch_aps = []
        
        for img_idx in range(len(preds)):
            img_preds = preds[img_idx]
            img_labels = labels[img_idx]
            
            # skip if no preds or labels
            if len(img_preds) == 0 or len(img_labels) == 0:
                continue

            ious = ops.box_iou(img_preds[:, 1:5], img_labels[:, 1:5])
            
            # sort preds by confidance 
            conf_sorted_indices = torch.argsort(img_preds[:, 0], descending=True)
            
            true_positives = torch.zeros(len(img_preds))
            false_positives = torch.zeros(len(img_preds))
            
            # track which true lables have been used
            used_gt = torch.zeros(len(img_labels))
            
            for pred_idx in conf_sorted_indices:
                best_iou, best_gt_idx = torch.max(ious[pred_idx], dim=0)
                
                if best_iou >= iou_thresh and not used_gt[best_gt_idx]:
                    true_positives[pred_idx] = 1
                    used_gt[best_gt_idx] = True
                else:
                    false_positives[pred_idx] = 1
            
            cum_tp = torch.cumsum(true_positives[conf_sorted_indices], dim=0)
            cum_fp = torch.cumsum(false_positives[conf_sorted_indices], dim=0)
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / len(img_labels)
            
            # add (precision=1 recall=0 )and (precision=0  recall=1) 
            precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
            recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
            
            # calc teh ap on the per img level
            ap = torch.trapezoid(precision, recall)
            batch_aps.append(ap.item())
        
        # mean across al imgs at this iou level
        if batch_aps:
            mAPs.append(sum(batch_aps) / len(batch_aps))
    
    return sum(mAPs) / len(mAPs)