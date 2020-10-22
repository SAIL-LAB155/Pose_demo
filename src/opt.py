import argparse
parser = argparse.ArgumentParser(description='Inner Configuration')


"----------------------------- Yolo options -----------------------------"
parser.add_argument('--confidence', default=None, type=float,
                    help='epoch of lr decay')
parser.add_argument('--num_classes', default=None, type=int,
                    help='epoch of lr decay')
parser.add_argument('--nms_thresh', default=None, type=float,
                    help='epoch of lr decay')
parser.add_argument('--input_size', default=None, type=int,
                    help='epoch of lr decay')


"----------------------------- Pose options -----------------------------"
parser.add_argument('--input_height', default=None, type=int,
                    help='epoch of lr decay')
parser.add_argument('--input_width', default=None, type=int,
                    help='epoch of lr decay')
parser.add_argument('--output_height', default=None, type=int,
                    help='epoch of lr decay')
parser.add_argument('--output_width', default=None, type=int,
                    help='epoch of lr decay')

parser.add_argument('--pose_weight', default=None, type=str,
                    help='epoch of lr decay')
parser.add_argument('--pose_backbone', default=None, type=str,
                    help='epoch of lr decay')
parser.add_argument('--pose_cfg', default=None, type=str,
                    help='epoch of lr decay')
parser.add_argument('--pose_cls', default=None, type=int,
                    help='epoch of lr decay')
parser.add_argument('--DUC_idx', default=None, type=int,
                    help='epoch of lr decay')
parser.add_argument('--pose_thresh', default=None, type=str,
                    help='epoch of lr decay')


"----------------------------- RNN options -----------------------------"
parser.add_argument('--RNN_backbone', default=None, type=float,
                    help='epoch of lr decay')
parser.add_argument('--TCN_single', default=False, type=float,
                    help='epoch of lr decay')
parser.add_argument('--RNN_frame_length', default=None, type=float,
                    help='epoch of lr decay')
parser.add_argument('--RNN_class', default=False, type=float,
                    help='epoch of lr decay')


"----------------------------- CNN options -----------------------------"
parser.add_argument('--CNN_class', default=0.05, type=float,
                    help='epoch of lr decay')
parser.add_argument('--CNN_backbone', default=2, type=float,
                    help='epoch of lr decay')
parser.add_argument('--CNN_thresh', default=2, type=float,
                    help='epoch of lr decay')


"----------------------------- Transfer options -------------------------"
parser.add_argument('--libtorch', default=None, type=int,
                    help='epoch of lr decay')


"----------------------------- Visualization options --------------------"
parser.add_argument('--plot_bbox', default=None, type=float,
                    help='epoch of lr decay')
parser.add_argument('--plot_kps', default=None, type=int,
                    help='epoch of lr decay')
parser.add_argument('--plot_id', default=None, type=float,
                    help='epoch of lr decay')
parser.add_argument('--track_id', default=None, type=int,
                    help='epoch of lr decay')
parser.add_argument('--track_plot_id', default=None, type=int,
                    help='epoch of lr decay')

opt, _ = parser.parse_known_args()
