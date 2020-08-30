import cv2
from src.utils.utils import cal_center_point
from src.utils.plot import thicks, sizes, colors


class AreaVisualizer:
    def __init__(self, w, h, w_num, h_num):
        self.height, self.width = h, w
        self.height_num, self.width_num = h_num, w_num
        self.h_interval, self.w_interval = int(h/h_num), int(w/w_num)
        pass

    def draw_boundary(self, img):
        for i in range(self.width_num - 1):
            cv2.line(img, (0, (i + 1) * self.h_interval),
                     (self.width, (i + 1) * self.h_interval), colors["yellow"], thicks["line"])
        for j in range(self.height_num - 1):
            cv2.line(img, ((j + 1) * self.w_interval, 0),
                     ((j + 1) * self.w_interval, self.height), colors["yellow"], thicks["line"])

    def draw_box(self, boxes, img):
        for box in boxes:
            [x1, y1, x2, y2] = box
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors["red"], thicks["box"])

    def draw_center_point(self, boxes, img):
        for box in boxes:
            cv2.circle(img, cal_center_point(box), sizes["point"], colors["cyan"], -1)

    def draw_cnt_map(self, img, REGIONS):
        for idx, region in REGIONS.items():
            cv2.putText(img, str(region.exists), region.center, cv2.FONT_HERSHEY_SIMPLEX, sizes["word"],
                        region.cnt_color(), thicks["word"])

    def draw_warning_mask(self, img, REGIONS, alarm_ls):
        # print(alarm_ls)
        for idx in alarm_ls:
            region = REGIONS[idx]
            img = cv2.rectangle(img, (region.left, region.top), (region.right, region.bottom), colors["red"], -1)
