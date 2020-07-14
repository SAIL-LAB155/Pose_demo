from .region import Region
import numpy as np
import math
import cv2
import copy
from src.utils.utils import cal_center_point
from .area_vis import AreaVisualizer


class RegionProcessor:
    def __init__(self, w, h, w_num, h_num, write=False):
        self.height, self.width = h, w
        self.height_num, self.width_num = h_num, w_num
        self.region_cnt = h_num * w_num
        self.h_interval, self.w_interval = int(h/h_num), int(w/w_num)
        self.region_idx = [(i, j) for i in range(h_num) for j in range(w_num)]
        self.REGIONS = {idx: Region(idx, self.h_interval, self.w_interval) for idx in self.region_idx}
        self.if_write = write
        self.Visualize = AreaVisualizer(w, h, w_num, h_num)

        if self.if_write:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter("output2.mp4", fourcc, 15, (w*2, h))

        self.keep_ls = []
        self.update_ls = []
        self.empty_ls = []
        self.alarm_ls = []
        self.img = np.array([[[]]])

    def clear(self):
        self.keep_ls = []
        self.update_ls = []
        self.empty_ls = []
        self.alarm_ls = []

    def locate(self, pt):
        return math.floor(pt[0]/self.w_interval), math.floor(pt[1]/self.h_interval)

    def locate_cover(self, pt_tl, pt_br):
        tl = self.locate(pt_tl)
        br = self.locate(pt_br)
        return [(i, j) for i in range(tl[0]+1, br[0]) for j in range(tl[1]+1, br[1])]

    def locate_occupy(self, pt_tl, pt_br):
        tl = self.locate(pt_tl)
        br = self.locate(pt_br)
        return [(i, j) for i in range(tl[0], br[0]+1) for j in range(tl[1], br[1]+1)]

    def center_region(self, boxes):
        center_region = []
        for box in boxes:
            center = cal_center_point(box)
            center_region.append(self.locate(center))
        return center_region

    def cover_region(self, boxes):
        cover_region = []
        for box in boxes:
            tl, br = (box[0], box[1]), (box[2], box[3])
            cover_range = self.locate_cover(tl, br)
            if cover_range:
                cover_region += cover_range
        return cover_region

    def occupy_region(self, boxes):
        occupy_region = []
        for box in boxes:
            tl, br = (box[0], box[1]), (box[2], box[3])
            occupy_range = self.locate_occupy(tl, br)
            if occupy_range:
                occupy_region += occupy_range
        return occupy_region

    def region_process(self, occupy, cover, center):
        for idx in self.REGIONS.keys():
            if idx in center or idx in cover:
                self.update_ls.append(idx)
            elif idx in occupy:
                self.keep_ls.append(idx)
            else:
                self.empty_ls.append(idx)

    def update_region(self):
        for i in self.update_ls:
            self.REGIONS[i].update(1)
        for i in self.empty_ls:
            self.REGIONS[i].update(-1)
        for i in self.keep_ls:
            self.REGIONS[i].update(0)

    def trigger_alarm(self, img):
        for idx, region in self.REGIONS.items():
            if region.if_warning():
                self.alarm_ls.append(idx)
        if self.alarm_ls:
            self.draw_alarm_signal(img)
            self.draw_warning_mask(img)

    def process_box(self, boxes, fr):
        self.clear()
        self.img = copy.deepcopy(fr)
        if boxes is not None:
            occupy_ls = self.occupy_region(boxes)
            cover_ls = self.cover_region(boxes)
            center_ls = self.center_region(boxes)
            self.region_process(occupy_ls, cover_ls, center_ls)
        else:
            self.empty_ls = self.region_idx
        self.update_region()
        self.trigger_alarm(fr)

        res = self.visualize(boxes, fr)
        cv2.imshow("result", res)
        if self.if_write:
            self.out.write(res)
        return res

    def draw_alarm_signal(self, img):
        cv2.putText(img, "HELP!!!", (360, 270), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

    def draw_cnt_map(self, img):
        for idx, region in self.REGIONS.items():
            cv2.putText(img, str(region.exists), region.center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, region.cnt_color(), 1)

    def draw_warning_mask(self, img):
        print(self.alarm_ls)
        for idx in self.alarm_ls:
            region = self.REGIONS[idx]
            img = cv2.rectangle(img, (region.left, region.top), (region.right, region.bottom), (0, 0, 255), -1)

    def visualize(self, boxes, img):
        im_black = cv2.imread("src/black.jpg")
        im_black = cv2.resize(im_black, (self.width, self.height))
        self.Visualize.draw_cnt_map(im_black, self.REGIONS)

        self.Visualize.draw_box(boxes, img)
        self.Visualize.draw_center_point(boxes, img)
        self.Visualize.draw_boundary(img)
        self.Visualize.draw_warning_mask(img, self.REGIONS, self.alarm_ls)
        res = np.concatenate((im_black, img), axis=1)
        return res


if __name__ == '__main__':
    boxes = [[103, 40, 470, 120], [60, 200, 160, 300], [310, 180, 520, 315]]

    im = cv2.imread("../black.jpg")
    RP = RegionProcessor(540, 360, 10, 10)
    im = cv2.resize(im, (540, 360))
    cnt_im = copy.deepcopy(im)

    RP.draw_boundary(im)
    RP.draw_box(boxes, im)
    RP.draw_center_point(boxes, im)
    centers = RP.center_region(boxes)
    covers = RP.cover_region(boxes)
    occupies = RP.occupy_region(boxes)

    cnt_im = RP.process_box(boxes, cnt_im)

    cv2.imshow("cnt", cnt_im)
    cv2.imshow("im", im)
    cv2.waitKey(0)

