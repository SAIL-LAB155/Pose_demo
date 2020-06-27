import time
import torch
from config.config import device


def get_inference_time(model, repeat=200, height=416, width=416):
    model.eval()
    start = time.time()
    with torch.no_grad():
        inp = torch.randn(1, 3, height, width)
        if device != "cpu":
            inp = inp.cuda()
        for i in range(repeat):
            output = model(inp)
    avg_infer_time = (time.time() - start) / repeat

    return round(avg_infer_time, 4)
