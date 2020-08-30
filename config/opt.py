import argparse

parser = argparse.ArgumentParser(description='Pose options')

parser.add_argument('--pose_weight', default='weights/sppe/duc_se.pth', type=str,
                    help='Pose weight location')
parser.add_argument('--pose_backbone', default="seresnet101", type=str,
                    help='backbone')
parser.add_argument('--cls_num', default=17, type=int,
                    help='sparse_decay')
parser.add_argument('--DUC_idx', default=0, type=int,
                    help='epoch of lr decay')
parser.add_argument('--pose_cfg', default=None, type=str,
                    help='The cfg')

parser.add_argument('--input_height', default=320, type=int,
                    help='input_height')
parser.add_argument('--input_width', default=256, type=int,
                    help='input_width')
parser.add_argument('--output_height', default=80, type=int,
                    help='output_height')
parser.add_argument('--output_width', default=64, type=str,
                    help='output_width')