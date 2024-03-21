import tensorrt as trt
from torch2trt import TRTModule
import torch
import time


logger = trt.Logger(trt.Logger.INFO)
# 加载engine
with open(r"C:\Users\yl\Desktop\大论\硬件测试\MobileTrack\backbone_255.engine", "rb") as f, trt.Runtime(
        logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

backbone_255 = TRTModule(engine, input_names=["input"], output_names=['output'])

with open(r"C:\Users\yl\Desktop\大论\硬件测试\MobileTrack\backbone_127.engine", "rb") as f, trt.Runtime(
        logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

backbone_127 = TRTModule(engine, input_names=["input"], output_names=['output'])

with open(r"C:\Users\yl\Desktop\大论\硬件测试\MobileTrack\head.engine", "rb") as f, trt.Runtime(
        logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

head = TRTModule(engine, input_names=["input1", "input2"], output_names=['output1', 'output2'])


if __name__ == '__main__':
    # ----------------------------- test -----------------------------------
    # fire the model
    for i in range(10):
        x = torch.randn(1, 3, 255, 255)

    inference_time = 0
    calls = 0
    start_time = time.time()

    # for inference time
    for i in range(1000):
        x = torch.randn(1, 3, 255, 255)
        calls += 1

    inference_time += time.time() - start_time
    print("average time = ", inference_time / calls)

