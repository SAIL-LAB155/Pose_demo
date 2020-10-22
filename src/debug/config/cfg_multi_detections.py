import torch

"------------------------------Outer configurations----------------------------"

video_path = "video/0619_115.mp4"

gray_yolo_cfg = "weights/yolo/gray/1010/yolov3-spp-1cls-leaky.cfg"
gray_yolo_weights = "weights/yolo/gray/1010/best.weights"
black_yolo_cfg = "weights/yolo/black/1010/yolov3-original-1cls-leaky.cfg"
black_yolo_weights = "weights/yolo/black/1010/best.weights"
rgb_yolo_cfg = ""
rgb_yolo_weights = ""

black_box_threshold = 0.5122
gray_box_threshold = 0.526

pose_weight = "weights/sppe/duc_se.pth"

water_top = 40

CNN_weight = "weights/CNN/underwater/1/1_mobilenet_9_decay1.pth"

RNN_weight = "weights/RNN/TCN_struct1_2020-07-08-20-02-32.pth"

store_frame = (3840, 2160)
resize_ratio = 0.5
show_size = (1440, 840)
store_size = (1440, 1080)


"------------------------------Inner configurations----------------------------"

device = "cuda:0"

# For yolo
confidence = 0.05
num_classes = 80
nms_thresh = 0.33
input_size = 416

# For pose estimation
input_height = 320
input_width = 256
output_height = 80
output_width = 64
fast_inference = True
pose_batch = 80

pose_backbone = "seresnet101"
pose_cls = 17
DUCs = [480, 240]
pose_cfg = None


libtorch = None
research = False

RNN_frame_length = 4
RNN_backbone = "TCN"
RNN_class = ["stand", "drown"]
TCN_single = True

import os
pose_option = os.path.join("/".join(pose_weight.replace("\\", "/").split("/")[:-1]), "option.pkl")
if os.path.exists(pose_option):
    info = torch.load(pose_option)
    pose_backbone = info.backbone
    pose_cfg = info.struct
    pose_cls = info.kps
    DUC_idx = info.DUC

    output_height = info.outputResH
    output_width = info.outputResW
    input_height = info.inputResH
    input_width = info.inputResW

CNN_class = ["drown", "floating", "standing"]
CNN_backbone = "mobilenet"
CNN_thresh = 0.95

import os
CNN_option = os.path.join("/".join(CNN_weight.replace("\\", "/").split("/")[:-1]), "option.pth")
if os.path.exists(CNN_option):
    info = torch.load(CNN_option)
    CNN_backbone = info.backbone
    CNN_class = info.classes.split(",")


from src.opt import opt

