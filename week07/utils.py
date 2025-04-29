import torch

def yolo_labels_to_xyxy(labels, S=7):
    #we assume imput size to be (BatchSize x S x S x 5)
    all_converted = []    
    for label in labels:
        batch_converted = []
        for row in range(S):
            for col in range(S):
                #if its a bbox then confidace will be 1
                if label[row][col][0] == 1: 
                    relative_x, relative_y = label[row][col][1], label[row][col][2]
                    relative_w, relative_h = label[row][col][3], label[row][col][4]

                    x_img = (col + relative_x) / S
                    y_img = (row + relative_y) / S
                    w_img = relative_w / S
                    h_img = relative_h / S
                    
                    x1 = x_img - w_img / 2
                    y1 = y_img - h_img / 2
                    x2 = x_img + w_img / 2
                    y2 = y_img + h_img / 2


                    batch_converted.append(torch.tensor([1, x1, y1, x2, y2]))
                        
        all_converted.append(torch.stack(batch_converted))
    return all_converted

def yolo_label_to_cxcywh(label, img_size, S=7, absolute_pixl_values=False):
    #we assume imput size to be (S x S x 5)
    all_converted = []
    img_w, img_h = img_size
    cell_w, cell_h =  img_w/S, img_h/S
    for row in range(S):
        for col in range(S):
            #if its a bbox then confidace will be 1
            if label[row][col][0] == 1: 
                relative_x, relative_y = label[row][col][1], label[row][col][2]
                relative_w, relative_h = label[row][col][3], label[row][col][4]

                #these are in absolute pxl values not 0-1
                true_x = (col + relative_x) * cell_w
                true_y = (row + relative_y) * cell_h
                true_w = relative_w * cell_w
                true_h = relative_h * cell_h

                if absolute_pixl_values:
                    all_converted.append(torch.tensor([1, true_x, true_y, true_w, true_h]))
                else:
                #to make it 0-1
                    all_converted.append(torch.tensor([1, true_x/img_w, true_y/img_h, true_w/img_w, true_h/img_h]))
                    
    return torch.stack(all_converted)