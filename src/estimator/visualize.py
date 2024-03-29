import torch
import cv2
from PIL import Image
import numpy as np
from config.config import pose_cls
from src.utils.plot import colors

RED = colors["red"]
GREEN = colors["green"]
BLUE = colors["blue"]
CYAN = colors["cyan"]
YELLOW = colors["yellow"]
ORANGE = colors["orange"]
PURPLE = colors["purple"]

if pose_cls == 13:
    coco_l_pair = [
        (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
        (13, 7), (13, 8), (0, 13),  # Body
        (7, 9), (8, 10), (9, 11), (10, 12)
    ]
    mpii_l_pair = [
        (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
        (13, 14), (14, 15), (3, 4), (4, 5),
        (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
    ]
elif pose_cls == 17:
    coco_l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    mpii_l_pair = [
        (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
        (13, 14), (14, 15), (3, 4), (4, 5),
        (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
    ]
else:
    raise ValueError("Wrong number")

coco_p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
           # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
           (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
coco_line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

mpii_p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
mpii_line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]


class KeyPointVisualizer(object):
    def __init__(self, format="coco"):
        if format == "coco":
            self.l_pair = coco_l_pair
            self.p_color = coco_p_color
            self.line_color = coco_line_color
        elif format == 'mpii':
            self.l_pair = mpii_l_pair
            self.p_color = mpii_p_color
            self.line_color = mpii_line_color
        else:
            raise NotImplementedError

    def __visualize(self, img, humans, scores, color):

        for idx in range(len(humans)):
            part_line = {}
            kp_preds = humans[idx]
            kp_scores = scores[idx]

            if pose_cls == 17:
                kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
                kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
            elif pose_cls == 13:
                kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[1, :] + kp_preds[2, :]) / 2, 0)))
                kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[1, :] + kp_scores[2, :]) / 2, 0)))

            # Draw keypoints
            for n in range(kp_scores.shape[0]):
                if kp_scores[n] <= 0.05:
                    continue
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                part_line[n] = (cor_x, cor_y)
                cv2.circle(img, (cor_x, cor_y), 4, self.p_color[n], -1)
            # Draw limbs
            for i, (start_p, end_p) in enumerate(self.l_pair):
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]
                    cv2.line(img, start_xy, end_xy, self.line_color[i], 8)

    def vis_ske(self, img, humans, scores):
        if isinstance(humans, dict):
            humans, scores = self.dict2ls(humans), self.dict2ls(scores)
        return self.__visualize(img, humans, scores, "origin")

    def vis_ske_black(self, img, humans, scores):
        if isinstance(humans, dict):
            humans, scores = self.dict2ls(humans), self.dict2ls(scores)
        return self.__visualize(img, humans, scores, "black")

    def dict2ls(self, d):
        return [torch.FloatTensor(v) for k, v in d.items()]

    def kpsdic2tensor(self, kps_dict, kpsScore_dict):
        ls_kp, ls_score = [], []
        for k, v in kps_dict.items():
            ls_kp.append(v)
            ls_score.append(kpsScore_dict[k])
        # ls = [v for k, v in d.items()]
        # score_temp = torch.FloatTensor([0.999]*pose_cls).unsqueeze(dim=1)
        return torch.FloatTensor(ls_kp), ls_score

    def scoredict2tensor(self, d):
        pass


