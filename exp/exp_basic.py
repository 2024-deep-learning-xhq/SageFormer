import os
import torch
from models import SageFormer
from models import Autoformer
from models import Informer
from models import Reformer
from models import Transformer
from models import PatchTST
from models import DLinear
from models import NLinear
from models import Linear

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SageFormer': SageFormer,
            'Autoformer': Autoformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Transformer': Transformer,
            'PatchTST': PatchTST,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
