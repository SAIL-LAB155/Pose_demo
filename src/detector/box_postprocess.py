import torch
from ..utils.img import cropBox, im_to_torch
# from config import config
import cv2
from src.yolo.bbox import bbox_iou
import numpy as np
from src.opt import opt

input_height = opt.input_height
input_width = opt.input_width


def crop_bbox(orig_img, boxes):
    with torch.no_grad():
        if orig_img is None:
            return None, None, None

        if boxes is None or boxes.nelement() == 0:
            return None, None, None

        inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        # inp = orig_img
        inps, pt1, pt2 = crop_from_dets(inp, boxes)
        return inps, pt1, pt2


def crop_from_dets(img, boxes):

    inps = torch.zeros(boxes.size(0), 3, input_height, input_width)
    pt1 = torch.zeros(boxes.size(0), 2)
    pt2 = torch.zeros(boxes.size(0), 2)

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor((float(box[0]), float(box[1])))
        bottomRight = torch.Tensor((float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, input_height, input_width)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight
    return inps, pt1, pt2


def filter_box(boxes, scores, res, thresh=0.3):
    keep_ls = []
    for idx, (b, s, r) in enumerate(zip(boxes, scores, res)):
        if s > thresh and right_distance(b[0], b[2]) and right_distance(b[1], b[3]):
            keep_ls.append(idx)
    return boxes[keep_ls], scores[keep_ls], res[keep_ls]


def right_distance(a, b):
    if abs(a-b) > 10:
        return True
    return False


def cal_area(box):
    area = (box[2]-box[0])*(box[3]-box[1])
    # print(area)
    return area


def nms(dets, conf=0.5):
    if len(dets) < 2:
        return dets

    max_detections = []
    while dets.size(0):
        # Get detection with highest confidence and save as max detection
        max_detections.append(dets[0].unsqueeze(0))
        # Stop if we're at the last detection
        if len(dets) == 1:
            break
        # Get the IOUs for all boxes with lower confidence
        ious = bbox_iou(max_detections[-1], dets[1:])
        # Remove detections with IoU >= NMS threshold
        dets = dets[1:][ious < conf]

    return torch.cat(max_detections)


def eliminate_nan(id2box):
    res = {}
    for k, v in id2box.items():
        if True not in np.isnan(v.numpy()):
            res[k] = v
    return res


class BoxEnsemble:
    def __init__(self, height, width):
        self.pre_boxes = []
        self.max_box = 1
        self.black_max_thresh = height * width * 0.4

    def ensemble_box(self, black_res, gray_res):
        black_keep = self.keep_small(black_res[:, :4], self.black_max_thresh)
        # gray_keep = self.analyse_area(gray_res[:,:4], "b")
        # merged_res = torch.cat((black_res[black_keep], gray_res[gray_keep]), dim=0)
        merged_res = torch.cat((black_res[black_keep], gray_res), dim=0)
        merged_res = nms(merged_res)
        return merged_res

    def keep_small(self, boxes, thresh):
        keep_idx = []
        if len(boxes) > 0:
            for idx, box in enumerate(boxes):
                if cal_area(box) < thresh:
                    keep_idx.append(idx)
        return keep_idx


