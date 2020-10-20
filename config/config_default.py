import torch

yolo_cfg = "config/yolo_cfg/yolov3-spp.cfg"
yolo_weight = 'weights/yolo/yolov3-spp.weights'
pose_weight = "weights/sppe/duc_se.pth"
pose_cfg = None

video_path = "video/video_sample/video4_Trim.mp4"

'''
------------------------------------------------------------------------------------------
'''

img_folder = "img/squat"

write_video = False
write_box = False
write_kps = False

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


frame_size = (544, 960)
libtorch = False

pose_backbone = "seresnet101"
pose_cls = 17
DUC_idx = 0
pose_thresh = [0.5] * 17

track_idx = "all"    # If all idx, track_idx = "all"
track_plot_id = ["all"]   # If all idx, track_plot_id = ["all"]
assert track_idx == "all" or isinstance(track_idx, int)

plot_bbox = True
plot_kps = True
plot_id = True


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

squat_up_side_video_path = "video/squat1/cyx_squat_side2.mp4"
