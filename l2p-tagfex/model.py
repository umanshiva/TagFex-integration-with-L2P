import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from .tagfexnet import TagFexNet
# from modules import Accuracy, MeanMetric, CatMetric, select_metrics, forward_metrics, get_metrics
# from modules import optimizer_dispatch, scheduler_dispatch, get_loaders
from utils.funcs import parameter_count

from loggers import LoguruLogger, loguru
from prompt import Prompt

EPSILON = 1e-8
from base import ContinualLearner

class Tagfexl2p(ContinualLearner):
    def __init__(self, data_manager, configs: dict, device, distributed=None) -> None:
        super().__init__(data_manager, configs, device, distributed)

        self.model = configs['model']
        self.original_model = configs['original_model']
        self.optimizer = configs['optimizer']
        self.scheduler = configs['scheduler']
        self.criterion = configs['criterion']
        self.args = configs['args']
    
