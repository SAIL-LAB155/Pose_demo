gray_yolo_cfg = "model/underwater_gray/yolov3-spp-1cls.cfg"
gray_yolo_weights = "model/underwater_gray/135_608_best.weights"
black_yolo_cfg = "model/underwater_black/yolov3-spp-1cls.cfg"
black_yolo_weights = "model/underwater_black/150_416_best.weights"
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

video_path = "video/underwater/0619_115.mp4"

black_box_threshold = 0.3
gray_box_threshold = 0.2
