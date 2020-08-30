from .people import Person
import cv2
from config.config import frame_size
import numpy as np
from src.utils.kp_process import KPSProcessor
from src.utils.plot import colors, sizes, thicks


class HumanProcessor:
    def __init__(self, width, height):
        self.stored_id = []
        self.PEOPLE = {}
        self.curr_id = []
        self.untracked_id = []
        self.curr_box_res = {}
        self.RD_warning = []
        self.RD_box_warning = []
        self.KPS_idx = []
        self.KPSP = KPSProcessor(height, width)

    def clear(self):
        self.curr_id = []
        self.untracked_id = []
        self.curr_box_res = {}
        self.RD_warning = []
        self.RD_box_warning = []
        self.KPS_idx = []

    def update_box(self, id2box):
        self.clear()
        for k, v in id2box.items():
            self.curr_id.append(k)
            if k not in self.stored_id:
                self.PEOPLE[k] = Person(k, v)
                self.stored_id.append(k)
            else:
                self.PEOPLE[k].BOX.append(v)
            self.curr_box_res[k] = self.PEOPLE[k].BOX.cal_size_ratio()
            self.PEOPLE[k].update_disappear(1)

    def update_untracked(self):
        self.untracked_id = [x for x in self.stored_id if x not in self.curr_id]
        for x in self.untracked_id:
            self.PEOPLE[x].update_disappear(0)

    def update(self, id2box):
        self.update_box(id2box)
        self.update_untracked()
        # self.vis_box_size()

    def box_size_warning(self, warning_ls):
        self.RD_warning = warning_ls
        self.RD_box_warning = [idx for idx in warning_ls if not self.PEOPLE[idx].BOX.get_size_ratio_info()]
        return self.RD_box_warning

    def vis_size_ls(self, im, idx, num):
        cv2.putText(im, "id{}".format(idx), (20 + 140*num, 40), cv2.FONT_ITALIC, 0.8, (0, 255, 255), 3)
        for i, item in enumerate(self.PEOPLE[idx].BOX.ratios.tolist()[::-1]):
            cv2.putText(im, "f{}: {}".format(i, round(item, 2)), (20 + 140*num, + 100+ 40*i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  colors[self.PEOPLE[idx].BOX.text_color(item)], 2)

    def vis_box_size(self, im_box, im_cnt):
        curr = sorted(self.curr_id)
        for num, idx in enumerate(curr):
            # self.PEOPLE[idx].BOX.vis_size(img_cnt, idx, num)
            self.vis_size_ls(im_cnt, idx, num)
            h, w = self.PEOPLE[idx].BOX.cal_curr_hw()
            cv2.putText(im_box, "{}".format(round((h/w).tolist(), 4)), self.PEOPLE[idx].BOX.curr_center(),
                        cv2.FONT_HERSHEY_SIMPLEX, sizes["word"], colors["cyan"], thicks["word"])

            tl, br = self.PEOPLE[idx].BOX.curr_box_location()
            if idx in self.RD_box_warning:
                cv2.putText(im_box, "id{}: Not standing".format(idx), self.PEOPLE[idx].BOX.curr_top(),
                        cv2.FONT_HERSHEY_SIMPLEX, sizes["word"], (255, 0, 255), thicks["word"])
                cv2.rectangle(im_box, tl, br, colors["violet"], thicks["box"])
            elif idx in self.RD_warning:
                cv2.putText(im_box, "id{}: Standing".format(idx), self.PEOPLE[idx].BOX.curr_top(),
                            cv2.FONT_HERSHEY_SIMPLEX, sizes["word"], (100, 255, 255), thicks["word"])
                cv2.rectangle(im_box, tl, br, colors["yellow"], thicks["box"])
            else:
                cv2.putText(im_box, "id{}".format(idx), self.PEOPLE[idx].BOX.curr_top(),
                            cv2.FONT_HERSHEY_SIMPLEX, sizes["word"], (255, 255, 255), thicks["word"])
                cv2.rectangle(im_box, tl, br, colors["white"], thicks["box"])

        im_box = cv2.resize(im_box, frame_size)
        return np.concatenate((im_cnt, im_box), axis=0)

    def update_kps(self, id2ske):
        for k, v in id2ske.items():
            coord = self.KPSP.process_kp(v)
            self.PEOPLE[k].KPS.append(coord)
            self.KPS_idx.append(k)

    def obtain_kps(self, idx):
        return self.PEOPLE[idx].KPS.kps

    def if_enough_kps(self, idx):
        return self.PEOPLE[idx].KPS.enough

    def update_RNN(self, idx, pred):
        self.PEOPLE[idx].update_RNN_pred(pred)

    def get_RNN_preds(self, idx):
        return self.PEOPLE[idx].RNN_preds
