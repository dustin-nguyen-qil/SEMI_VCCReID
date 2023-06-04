from typing import Any
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from torch import nn, optim
import numpy as np
import torch
from torch.optim import lr_scheduler
from config import CONFIG
from datasets.dataset_loader import build_dataloader
from models import build_models
from utils.losses import build_losses
from utils.utils import get_logger, save_checkpoint
from torchmetrics import functional as FM
from models.vid_resnet import C2DResNet50

class Baseline(LightningModule):
    def __init__(self) -> None:
        super(Baseline).__init__()

        self.trainloader, self.queryloader, self.galleryloader, self.dataset, self.train_sampler \
            = build_dataloader()
        
        # pid2clothes = torch.from_numpy(self.dataset.pid2clothes)

        # Build model
        self.appearance_model, self.shape_model, self.fusion_net, self.identity_classifier, self.clothes_classifier = build_models(
            CONFIG, self.dataset.num_pids, self.dataset.num_clothes)
        # Build identity classification loss, pairwise loss, clothes classificaiton loss
        # and adversarial loss.
        self.criterion_cla, self.criterion_pair, self.criterion_clothes, self.criterion_adv, self.shape_loss = build_losses(
            CONFIG, self.dataset.num_clothes)

        self.training_step_outputs = []
        
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
    
    def on_train_epoch_start(self) -> None:
        self.train_sampler.set_epoch(self.current_epoch)
    
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
        imgs, pids, camids, clothes_ids, xcs, betas = batch 
        appearance_feature = self.app_forward(imgs)
        if CONFIG.TRAIN.WITH_SHAPE:
            shape_feature = self.shape_forward(xcs)
            fused_feature = self.fusion(appearance_feature, shape_feature)
            features = fused_feature
        else:
            features = appearance_feature

        logits = self.identity_classifier(features)
        id_loss = self.criterion_cla(logits, pids)
        pair_loss = self.criterion_pair(features, pids)

        loss = id_loss + 0.5*pair_loss
        acc = FM.accuracy(logits, pids, 'multiclass', average='macro', num_classes=self.dataset.num_pids)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_step_outputs.append(loss)
        return loss 
    
    def on_train_epoch_end(self):
        epoch_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
        self.log('epoch_loss', epoch_loss)
        self.training_step_outputs.clear()

class Inference(nn.Module):
    def __init__(self, config) -> None:
        super(Inference).__init__()
        self.model = C2DResNet50(config)
    
    def forward(self, imgs):
        return self.model(imgs)

