from src.estimator.visualize import KeyPointVisualizer
from src.estimator.nms import pose_nms
from src.estimator.datatset import Mscoco
from config import config
import cv2
from utils.cal_time import get_inference_time
import torch
from utils.compute_flops import print_model_param_flops
from config.config import plot_kps, device
from utils.eval import getPrediction


class PoseEstimator(object):
    def __init__(self):
        self.skeleton = []
        self.KPV = KeyPointVisualizer()
        pose_dataset = Mscoco()
        # if config.fast_inference:
        #     self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        # else:
        #     self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        if config.pose_backbone == "seresnet101":
            from models.seresnet.FastPose import InferenNet_fast as createModel
            self.pose_model = createModel(4 * 1 + 1, pose_dataset, cfg=config.pose_cfg)
        elif config.pose_backbone == "mobilenet":
            from models.mobilenet.MobilePose import createModel
            self.pose_model = createModel(cfg=config.pose_cfg)
        else:
            raise ValueError("Not a backbone!")
        if config.device != "cpu":
            self.pose_model.cuda()
            self.pose_model.eval()
        inf_time = get_inference_time(self.pose_model, height=config.input_height, width=config.input_width)
        print("The average inference time of detection is {}s".format(inf_time))
        self.batch_size = config.pose_batch
        # flops = print_model_param_flops(self.pose_model)
        # print("The flops of current pose estimation model is {}".format(flops))

    def process_img(self, inps, orig_img, boxes, scores, pt1, pt2):
        # try:
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % self.batch_size:
                leftover = 1
            num_batches = datalen // self.batch_size + leftover
            hm = []

            for j in range(num_batches):
                if device != "cpu":
                    inps_j = inps[j * self.batch_size:min((j + 1) * self.batch_size, datalen)].cuda()
                else:
                    inps_j = inps[j * self.batch_size:min((j + 1) * self.batch_size, datalen)]
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm).cpu().data

            preds_hm, preds_img, preds_scores = getPrediction(
                hm, pt1, pt2, config.input_height, config.input_width, config.output_height, config.output_width)
            kps, kps_score = pose_nms(boxes, scores, preds_img, preds_scores)

            return kps, kps_score
        # except:
        #     return [], orig_img, cv2.imread("video/black.jpg")
