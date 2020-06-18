import numpy as np
import math
from collections import defaultdict
import cv2
import torch
from config.config import pose_cls

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]


class Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def get_angle(center_coor, coor2, coor3):
        L1 = Utils.cal_dis(coor2,coor3)
        L2 = Utils.cal_dis(center_coor,coor3)
        L3 = Utils.cal_dis(center_coor,coor2)
        Angle = Utils.cal_angle(L1,L2,L3)
        return Angle

    @staticmethod
    def cal_dis(coor1, coor2):
        out = np.square(coor1[0] - coor2[0]) + np.square(coor1[1] - coor2[1])
        return np.sqrt(out)

    @staticmethod
    def cal_angle(L1, L2, L3):
        out = (np.square(L2) + np.square(L3) - np.square(L1)) / (2 * L2 * L3)
        try:
            return math.acos(out) * (180 / math.pi)
        except ValueError:
            return 180

    @staticmethod
    def count_average(origin_ls):
        length = len(origin_ls)
        if length != 0:
            x_all = 0
            y_all = 0
            for i in range(int(length/2)):
                x_all += origin_ls[2*i]
                y_all += origin_ls[2*i +1]
            x_ave = x_all*2 / length
            y_ave = y_all*2 / length
            return [x_ave, y_ave]
        else:
            return []

    @staticmethod
    def image_normalize(image, size=224):
        image_array = cv2.resize(image, (size, size))
        image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
        image_array = image_array.transpose((2, 0, 1))
        for channel, _ in enumerate(image_array):
            image_array[channel] /= 255.0
            image_array[channel] -= image_normalize_mean[channel]
            image_array[channel] /= image_normalize_std[channel]
        image_tensor = torch.from_numpy(image_array).float()
        return image_tensor


def box2str(box):
    sub_box = ""
    for coor in box:
        sub_box += str(coor)
        sub_box += ","
    return sub_box[:-1]


def kp2str(kp):
    sub_kp = ""
    for item in kp:
        sub_kp += str(item[0])
        sub_kp += ","
        sub_kp += str(item[1])
        sub_kp += ","
    return sub_kp[:-1]


def kpScore2str(scores):
    scores = scores.tolist()
    sub_s = ""
    for item in scores:
        sub_s += str(item[0])
        sub_s += ","
    return sub_s[:-1]


def str2boxdict(s):
    d = defaultdict(list)
    id_bboxs = s.split("\t")
    for item in id_bboxs[:-1]:
        [idx, box] = item.split(":")
        bbox = box.split(",")
        d[idx] = [int(float(b)) for b in bbox]
    return d


def str2kpsdict(s):
    d = defaultdict(list)
    id_kps = s.split("\t")
    for item in id_kps[:-1]:
        [idx, rawkps] = item.split(":")
        kps_ls, kps = rawkps.split(","), []
        for i in range(pose_cls):
            kps.append([float(kps_ls[i*2]), float(kps_ls[i*2+1])])
        d[idx] = kps
    return d

def boxdict2str(k, v):
    boxstr = box2str(v)
    return "{}:{}\t".format(str(k), boxstr)

def kpsdict2str(k,v):
    kpstr = kp2str(v)
    return "{}:{}\t".format(str(k), kpstr)


def str2box(string):
    if string == "":
        return None
    tmp = string.split(",")
    boxes = []
    for item in tmp:
        boxes.append([float(i) for i in item.split(" ")])
    return boxes

def str2kps(string):
    if string == "":
        return None
    tmp = string.split(",")
    boxes = []
    for item in tmp:
        boxes.append([float(i) for i in item.split(" ")])
    return boxes


def kpsScoredict2str(k,v):
    kpstr = kpScore2str(v)
    return "{}:{}\t".format(str(k), kpstr)


def str2kpsScoredict(s):
    d = defaultdict()
    id_kps = s.split("\t")
    for item in id_kps[:-1]:
        [idx, raw_score] = item.split(":")
        tmp = [[float(item)] for item in raw_score.split(",")]
        score = torch.FloatTensor(tmp)
        d[idx] = score
    return d


def gray3D(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


if __name__ == '__main__':
    ut = Utils()
    # res = ut.time_to_string("10.0000")
    # print(res)
    res = ut.get_angle([0, 0], [1, -1], [0, 1])
    print(res)
