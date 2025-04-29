from torch.utils.benchmark import Timer
import sys
import os
from torchvision import transforms
from PIL import Image
from model import FaceDetector
import torch

def benchmark_single_img(model, img, runs=100, device='cuda:0'):
    model.eval()

    model = model.to(device)
    img = img.to(device)

    #does the warm up by its self
    t = Timer(stmt='model(img)',
    globals={'model': model, 'img': img}, 
    num_threads=1)

    res = t.timeit(runs)

    print(f"Average time per image: {res.median * 1000:.3f} ms over {runs} runs")
    print(res)

def main():
    path_to_experiments_dir = sys.argv[1]
    img_path = "/home/kuba/Documents/data/raw/face-detection-dataset/images/train/ffff54ccbcad1111.jpg"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((448,448), antialias=True),  # Resize all images to same dimensions
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(dim=0)

    for dir in sorted(os.listdir(path_to_experiments_dir)):
        if dir == "00":
            model_dict_path = os.path.join(path_to_experiments_dir, dir, 'best_dev_loss057_dsc.pth')
        elif dir == "01":
            model_dict_path = os.path.join(path_to_experiments_dir, dir, 'best_dev_loss053_dsc.pth')
        elif dir == "02":
            model_dict_path = os.path.join(path_to_experiments_dir, dir, 'best_dev_loss0476_dsc.pth')
        print(f"Res for: {model_dict_path.split()[-1]}")

        model = FaceDetector()
        model.load_state_dict(state_dict=torch.load(model_dict_path, weights_only=True))

        benchmark_single_img(model, img_tensor, runs=100, device='cpu')
        
    

if __name__ == "__main__":
    main() 