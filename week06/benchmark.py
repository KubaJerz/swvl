from torch.utils.benchmark import Timer
import sys
import os

def benchmark_single_img(model, img, runs=100, device='cuda:0'):
    model.eval()

    model = model.to(device)
    img = img.to(device)

    #does the warm up by its self
    t = Timer(stmt='model(img)',
    setup='from __main__ import model, img',
    num_threads=1)

    res = t.timeit(runs)

    print(f"Average time per image: {res.median * 1000:.3f} ms over {runs} runs")
    print(res)

def main():
    path_to_experiments_dir = sys.argv[1]

    for dir in sorted(os.listdir(path_to_experiments_dir)):
        
    

if __name__ == "__main__":
    main()