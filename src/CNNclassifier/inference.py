# -*- coding:utf-8 -*-
from __future__ import print_function
from config.config import device, input_size
from .models import CNNModel, LeNet
import torch
import numpy as np
from torch import nn
from src.utils.utils import image_normalize
from src.opt import opt
import cv2
from src.utils.plot import colors, sizes, thicks

CNN_class = opt.CNN_class
CNN_backbone = opt.CNN_backbone
CNN_weight = opt.CNN_weight

onnx = opt.onnx
libtorch = opt.libtorch


class CNNInference(object):
    def __init__(self, class_nums=len(CNN_class), pre_train_name=CNN_backbone, model_path=CNN_weight):
        self.CNN_class = CNN_class
        if pre_train_name != "LeNet":
            self.model = CNNModel(class_nums, pre_train_name, model_path).model.to(device)
        else:
            self.model = LeNet(class_nums)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        if libtorch:
            example = torch.rand(2, 3, 224, 224).cuda()
            traced_model = torch.jit.trace(self.model, example)
            traced_model.save("CNN_lib.pt")

    def predict_result(self, img):
        img_tensor_list = []
        img_tensor = image_normalize(img, size=input_size)
        img_tensor_list.append(torch.unsqueeze(img_tensor, 0))
        if len(img_tensor_list) > 0:
            input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
            res_array = self.__inference(input_tensor)
            return res_array
        return None

    def predict_idx(self, img):
        out = self.predict_result(img)
        idx = out[0].tolist().index(max(out[0].tolist()))
        return idx

    def predict_class(self, img):
        idx = self.predict_idx(img)
        pred = self.CNN_class[idx]
        return pred

    def __inference(self, image_batch_tensor):
        self.model.eval()
        image_batch_tensor = image_batch_tensor.cuda()
        outputs = self.model(image_batch_tensor)
        outputs_tensor = outputs.data
        m_softmax = nn.Softmax(dim=1)
        outputs_tensor = m_softmax(outputs_tensor).to("cpu")
        return np.asarray(outputs_tensor)

    def classify(self, fr, id2bbox):
        pred_res = {}
        for idx, box in id2bbox.items():
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1 = 0 if x1 < 0 else x1
            y1 = 0 if y1 < 0 else y1
            x2 = fr.shape[1] if x2 > fr.shape[1] else x2
            y2 = fr.shape[0] if y2 > fr.shape[0] else y2
            img = np.asarray(fr[y1:y2, x1:x2])

            pred_array = self.predict_result(img)
            idx = pred_array[0].tolist().index(max(pred_array[0].tolist()))
            prediction = self.CNN_class[idx]

            text_location = (int((box[0]+box[2])/2)), int((box[1])+50)
            pred_res[text_location] = (prediction, idx, pred_array)
        return pred_res

    def visualize(self, frame, predictions):
        for i, (location, (pred, idx, score)) in enumerate(predictions.items()):
            cv2.putText(frame, "id{}: {}".format(idx, pred), location, cv2.FONT_HERSHEY_SIMPLEX, sizes["word"],
                        colors["red"], thicks["word"])
            text = self.array2str(score)
            cv2.putText(frame, "id{}: {}".format(idx, text), (30, 30 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, sizes["word"],
                        colors["yellow"], thicks["word"])

    @staticmethod
    def array2str(array):
        base = str(np.round(array[0], 4))
        for item in array[1:]:
            base += ","
            base += str(np.round(item, 4))
        return base[:-1]
