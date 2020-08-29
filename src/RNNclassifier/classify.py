from .BiLSTM.test_BiLSTM import BiLSTMPredictor
from .ConvGRU.test_ConvGRU import ConvGRUPredictor
from .TCN.test_TCN import TCNPredictor
from .ConvLSTM.test_ConvLstm import ConvLSTMPredictor
try:
    from .LSTM.test_LSTM import LSTMPredictor
    lstm = True
except:
    lstm = False

import numpy as np
from src.debug.config.cfg_with_RNN import RNN_backbone, RNN_class, RNN_weight


class RNNInference:
    def __init__(self, model_path=RNN_weight):
        self.tester = self.__get_tester(model_path)

    def __get_tester(self, model):
        if "ConvLSTM" == RNN_backbone:
            return ConvLSTMPredictor(model, len(RNN_class))
        if "BiLSTM" == RNN_backbone:
            return BiLSTMPredictor(model, len(RNN_class))
        if "ConvGRU" == RNN_backbone:
            return ConvGRUPredictor(model, len(RNN_class))
        if 'LSTM' == RNN_backbone:
            if lstm:
                return LSTMPredictor(model)
            else:
                print("lstm is not usable")
        if "TCN" == RNN_backbone:
            return TCNPredictor(model, len(RNN_class))

    def predict(self, inp):
        pred = self.tester.predict(np.array(inp).astype(np.float32))
        return pred
