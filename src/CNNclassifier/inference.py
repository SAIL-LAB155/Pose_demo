# -*- coding:utf-8 -*-
from __future__ import print_function
from config.config import device, input_size
from .models import CNNModel, LeNet
import torch
import numpy as np
from torch import nn
from src.utils.utils import image_normalize

from config.config import CNN_backbone, CNN_class, CNN_weight


class CNNInference(object):
    def __init__(self, class_nums=len(CNN_class), pre_train_name=CNN_backbone, model_path=CNN_weight):
        if pre_train_name != "LeNet":
            self.model = CNNModel(class_nums, pre_train_name, model_path).model.to(device)
        else:
            self.model = LeNet(class_nums)
        self.model.load_state_dict(torch.load(model_path, map_location=device))

    def predict(self, img):
        img_tensor_list = []
        img_tensor = image_normalize(img, size=input_size)
        img_tensor_list.append(torch.unsqueeze(img_tensor, 0))
        if len(img_tensor_list) > 0:
            input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
            res_array = self.__predict_image(input_tensor)
            return res_array
        return None

    def __predict_image(self, image_batch_tensor):
        self.model.eval()
        image_batch_tensor = image_batch_tensor.cuda()
        outputs = self.model(image_batch_tensor)
        outputs_tensor = outputs.data
        m_softmax = nn.Softmax(dim=1)
        outputs_tensor = m_softmax(outputs_tensor).to("cpu")
        return np.asarray(outputs_tensor)
