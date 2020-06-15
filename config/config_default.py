import torch

video_path = "video/new_video/single_half.mp4"
img_folder = "img/test"

write_video = False
write_box = True
write_kps = True

device = "cuda:0"
print("Using {}".format(device))

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


frame_size = (720,540)

pose_backbone = "seresnet101"
pose_weight = "weights/sppe/duc_se.pth"
pose_cfg = None
pose_cls = 17

DUCs = [480, 240]


yolo_cfg = "config/yolo_cfg/yolov3.cfg"
yolo_model = 'weights/yolo/yolov3.weights'


track_idx = "all"    # If all idx, track_idx = "all"
track_plot_id = ["all"]   # If all idx, track_plot_id = ["all"]
assert track_idx == "all" or isinstance(track_idx, int)

plot_bbox = True
plot_kps = True
plot_id = True
