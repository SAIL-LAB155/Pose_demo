gray_yolo_cfg = "model/gray/1007/13/yolov3-spp-1cls-leaky.cfg"
gray_yolo_weights = "model/gray/1007/13/best.weights"
black_yolo_cfg = "model/black/1007/2/yolov3-original-1cls-leaky.cfg"
black_yolo_weights = "model/black/1007/2/best.weights"
rgb_yolo_cfg = ""
rgb_yolo_weights = ""

pose_weight = "../../weights/sppe/duc_se.pth"
pose_cfg = None
pose_cls = 17

RNN_frame_length = 4
RNN_backbone = "TCN"
RNN_class = ["stand", "drown"]
RNN_weight = "model/RNN/TCN_struct1_2020-07-08-20-02-32.pth"
TCN_single = True

video_path = "video/underwater/vlc-record-2020-07-03-11h28m47s-1.avi-.mp4"

black_box_threshold = 0.7
gray_box_threshold = 0.7
