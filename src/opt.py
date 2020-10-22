import argparse
parser = argparse.ArgumentParser(description='Inner Configuration')


"----------------------------- Yolo options -----------------------------"
parser.add_argument('--confidence', default=0.05, type=float,
                    help='epoch of lr decay')
parser.add_argument('--num_classes', default=80, type=int,
                    help='epoch of lr decay')
parser.add_argument('--nms_thresh', default=0.33, type=float,
                    help='epoch of lr decay')
parser.add_argument('--input_size', default=416, type=int,
                    help='epoch of lr decay')


"----------------------------- Image Process options ---------------------"
parser.add_argument('--water_top', default=40, type=int,
                    help='epoch of lr decay')


"----------------------------- Pose options ------------------------------"
parser.add_argument('--input_height', default=320, type=int,
                    help='epoch of lr decay')
parser.add_argument('--input_width', default=256, type=int,
                    help='epoch of lr decay')
parser.add_argument('--output_height', default=80, type=int,
                    help='epoch of lr decay')
parser.add_argument('--output_width', default=64, type=int,
                    help='epoch of lr decay')

parser.add_argument('--pose_batch', default=80, type=int,
                    help='epoch of lr decay')
parser.add_argument('--fast_inference', default=True, type=bool,
                    help='epoch of lr decay')

parser.add_argument('--pose_weight', default="weights/sppe/duc_se.pth", type=str,
                    help='epoch of lr decay')
parser.add_argument('--pose_backbone', default="seresnet101", type=str,
                    help='epoch of lr decay')
parser.add_argument('--pose_cfg', default=None, type=str,
                    help='epoch of lr decay')
parser.add_argument('--pose_cls', default=17, type=int,
                    help='epoch of lr decay')
parser.add_argument('--DUC_idx', default=0, type=int,
                    help='epoch of lr decay')
parser.add_argument('--pose_thresh', default=[], type=list,
                    help='epoch of lr decay')


"----------------------------- RNN options ------------------------------"
parser.add_argument('--RNN_backbone', default="", type=str,
                    help='epoch of lr decay')
parser.add_argument('--TCN_single', default=False, type=bool,
                    help='epoch of lr decay')
parser.add_argument('--RNN_frame_length', default=4, type=int,
                    help='epoch of lr decay')
parser.add_argument('--RNN_class', default=2, type=int,
                    help='epoch of lr decay')


"----------------------------- CNN options ------------------------------"
parser.add_argument('--CNN_class', default=[], type=list,
                    help='epoch of lr decay')
parser.add_argument('--CNN_backbone', default="mobilenet", type=str,
                    help='epoch of lr decay')
parser.add_argument('--CNN_weight', default="", type=str,
                    help='epoch of lr decay')
parser.add_argument('--CNN_thresh', default=0.5, type=float,
                    help='epoch of lr decay')


"----------------------------- Convert options ---------------------------"
parser.add_argument('--libtorch', default="", type=str,
                    help='epoch of lr decay')
parser.add_argument('--onnx', default="", type=str,
                    help='epoch of lr decay')


"----------------------------- Visualization options ---------------------"
parser.add_argument('--plot_bbox', default=True, type=bool,
                    help='epoch of lr decay')
parser.add_argument('--plot_kps', default=True, type=bool,
                    help='epoch of lr decay')
parser.add_argument('--plot_id', default=True, type=bool,
                    help='epoch of lr decay')
parser.add_argument('--track_id', default="all", type=str,
                    help='epoch of lr decay')
parser.add_argument('--track_plot_id', default="all", type=list,
                    help='epoch of lr decay')

"----------------------------- Other options ----------------------------"
parser.add_argument('--device', default="cuda:0", type=str,
                    help='epoch of lr decay')

opt, _ = parser.parse_known_args()
