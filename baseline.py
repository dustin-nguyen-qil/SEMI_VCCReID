from typing import Any
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from torch import nn, optim
import numpy as np
import torch
from torch.optim import lr_scheduler
from cfg.model_1 import CONFIG
from datasets import build_dataloader
from models import build_models
from train import train
from utils.losses import build_losses
from utils.utils import get_logger, save_checkpoint

class Baseline(LightningModule):
    def __init__(self) -> None:
        super(Baseline).__init__()

        self.trainloader, self.queryloader, self.galleryloader, self.dataset, self.train_sampler \
            = build_dataloader(CONFIG)
        
        pid2clothes = torch.from_numpy(self.dataset.pid2clothes)

        # Build model
        self.appearance_model, self.shape_model, self.fusion_net, self.identity_classifier, self.clothes_classifier = build_models(
            CONFIG, self.dataset.num_train_pids, self.dataset.num_train_clothes)
        # Build identity classification loss, pairwise loss, clothes classificaiton loss
        # and adversarial loss.
        self.criterion_cla, self.criterion_pair, self.criterion_clothes, self.criterion_adv = build_losses(
            CONFIG, self.dataset.num_train_clothes)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=CONFIG.TRAIN.OPTIMIZER.LR,
            weight_decay=CONFIG.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=CONFIG.TRAIN.LR_SCHEDULER.STEPSIZE,
            gamma=CONFIG.TRAIN.LR_SCHEDULER.DECAY_RATE)
        
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return self.trainloader
    
    def app_forward(self, images):
        appearance_feature = self.appearance_model(images)
        return appearance_feature
    
    def shape_forward(self, xcs):
        shape_feature = self.shape_model(xcs)
        return shape_feature
    
    def fusion(self, appearance_feature, shape_feature):
        final_feature = self.fusion_net(appearance_feature, shape_feature)
        return final_feature
    
    def training_step(self, batch, batch_idx):
        imgs, pids, camids, clothes_ids = batch 
        appearance_feature = self.app_forward(imgs)
        shape_feature = self.shape_forward()
        return 

