from .visualize import KeyPointVisualizer
from .nms import pose_nms
from .datatset import Mscoco
from src.opt import opt
from ..utils.benchmark import *
import torch
from ..utils.eval import getPrediction

pose_backbone = opt.pose_backbone
pose_cfg = opt.pose_cfg
input_height = opt.input_height
input_width = opt.input_width
output_height = opt.output_height
output_width = opt.output_width

pose_batch = opt.pose_batch
libtorch = opt.libtorch
device = opt.device
onnx = opt.onnx


class PoseEstimator:
    def __init__(self, pose_cfg, pose_weight):
        self.KPV = KeyPointVisualizer()
        pose_dataset = Mscoco()

        if pose_backbone == "seresnet101":
            from src.pose_model.seresnet.FastPose import InferenNet_fast as createModel
            self.pose_model = createModel(4 * 1 + 1, pose_dataset, pose_weight, cfg=pose_cfg)
        elif pose_backbone == "mobilenet":
            from src.pose_model.mobilenet.MobilePose import createModel
            self.pose_model = createModel(cfg=pose_cfg)
            self.pose_model.load_state_dict(torch.load(pose_weight, map_location=device))
        else:
            raise ValueError("Not a backbone!")
        if device != "cpu":
            self.pose_model.cuda()
            self.pose_model.eval()
        inf_time = get_inference_time(self.pose_model, height=input_height, width=input_width)
        flops = print_model_param_flops(self.pose_model)
        params = print_model_param_nums(self.pose_model)
        print("Pose estimation: Inference time {}s, Params {}, FLOPs {}".format(inf_time, params, flops))
        if libtorch:
            example = torch.rand(2, 3, 256, 320)
            traced_model = torch.jit.trace(self.pose_model, example)
            traced_model.save("pose_lib.pt")
        with torch.no_grad():
            torch_out = torch.onnx.export(self.pose_model, torch.rand(1, 3, 256, 320).cuda(), "onnx_pose.onnx", verbose=False,)
        self.batch_size = pose_batch

    def process_img(self, inps, boxes, pt1, pt2):
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

        preds_hm, preds_img, preds_scores = getPrediction(hm, pt1, pt2, input_height, input_width, output_height,
                                                          output_width)
        kps, kps_score, kps_id = pose_nms(boxes, preds_img, preds_scores)

        return kps, kps_score, kps_id

