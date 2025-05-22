import cv2
from matplotlib import pyplot as plt
from nmsupr import non_max_supp
from model import FaceDetector
import torch
from torchvision import transforms

cam = cv2.VideoCapture(0)

PATH_TO_MODEL="/Users/kuba/Desktop/best_dev_loss0476_dsc.pth"

device = "mps"

model = FaceDetector()
model.load_state_dict(torch.load(PATH_TO_MODEL, weights_only=True, map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((448,448), antialias=True),  # Resize all images to same dimensions
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

while True:
    ret, frame = cam.read()
    height, width = frame.shape[:2]

    frame_transform = transform(frame)
    frame_transform = frame_transform.unsqueeze(dim=0)
    frame_transform = frame_transform.to(device)
    preds = model(frame_transform)
    preds = preds.cpu()
    preds = preds.view(-1, 7, 7, 10)
    #can mess with confidence and iou thresholds for different results
    found = non_max_supp(preds, confidence_threshold = 0.5, iou_threshold=0.2, S=7, return_type='xyxy')

    for face in found[0]:
        conf, x1, y1, x2, y2 = face
        x, y, w, h = x1*width, y1*height, x2*width, y2*height

        cv2.rectangle(frame, (int(x), int(y)), (int(w) , int(h)), (0, 0, 255), 5)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
