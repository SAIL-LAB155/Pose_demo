import torch
import os


"-------------------Outer configuration-----------------------"

yolo_cfg = "config/yolo_cfg/yolov3.cfg"
yolo_weight = "weights/yolo/yolov3.weights"
video_path = "video/video_sample/video4_Trim.mp4"
img_folder = "img/yoga_demo_whole"

pose_weight = "weights/sppe/yoga_mobile_1/1_best_acc.pkl"

write_video = False
write_box = False
write_kps = False

resize_ratio = 0.5
show_size = (1080, 720)
store_size = (1080, 720)


"-------------------Inner configuration-----------------------"

device = "cuda:0"
print("Using {}".format(device))

confidence = 0.8
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

onnx = True
libtorch = False

pose_cfg = None
pose_backbone = "seresnet101"
pose_cls = 17
DUC_idx = 0
pose_thresh = [0.05] * pose_cls
pose_thresh.append((pose_thresh[-11] + pose_thresh[-12]) / 2)

track_idx = "all"    # If all idx, track_idx = "all"
track_plot_id = ["all"]   # If all idx, track_plot_id = ["all"]
assert track_idx == "all" or isinstance(track_idx, int)

plot_bbox = True
plot_kps = True
plot_id = True

pose_option = os.path.join("/".join(pose_weight.replace("\\", "/").split("/")[:-1]), "option.pkl")
if os.path.exists(pose_option):
    info = torch.load(pose_option)
    pose_backbone = info.backbone
    pose_cfg = info.struct
    pose_cls = info.kps
    DUC_idx = info.DUC
    try:
        pose_thresh = list(map(lambda x: float(x), info.thresh.split(",")))
        pose_thresh.append((pose_thresh[-11] + pose_thresh[-12])/2)
    except:
        pass

    output_height = info.outputResH
    output_width = info.outputResW
    input_height = info.inputResH
    input_width = info.inputResW

"----------------------------Set inner configuration with opt-------------------------"

from src.opt import opt

opt.device = device

opt.output_height = output_height
opt.output_width = output_width
opt.input_height = input_height
opt.input_width = input_width

opt.pose_backbone = pose_backbone
opt.pose_cfg = pose_cfg
opt.pose_cls = pose_cls
opt.DUC_idx = DUC_idx
opt.pose_thresh = pose_thresh

opt.fast_inference = fast_inference
opt.pose_batch = pose_batch

opt.confidence = confidence
opt.num_classes = num_classes
opt.nms_thresh = nms_thresh
opt.input_size = input_size

opt.libtorch = libtorch
opt.onnx = onnx

opt.plot_bbox = plot_bbox
opt.plot_kps = plot_kps
opt.plot_id = plot_id
opt.track_id = track_idx
opt.track_plot_id = track_plot_id


