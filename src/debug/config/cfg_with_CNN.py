CNN_weight = "model/CNN/golf_ske_shufflenet_2019-10-14-08-36-18.pth"
CNN_backbone = "shufflenet"
CNN_class = ["Backswing", "FollowThrough", "Standing"]

yolo_cfg = "../../config/yolo_cfg/yolov3.cfg"
yolo_weight = "../../weights/yolo/yolov3.weights"

pose_weight = "../../weights/sppe/duc_se.pth"
pose_cfg = None

video_path = "video/withCNN/00.avi"
