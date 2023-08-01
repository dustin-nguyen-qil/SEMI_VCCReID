from pytorch_lightning import LightningModule 
from torch import nn, optim
from torch.optim import lr_scheduler 
from torchmetrics import functional as FM
import torch
from config import CONFIG
from datasets.dataset_loader import build_trainloader
from models import build_models
from utils.losses import build_losses, compute_loss
from models.texture.vid_resnet import *


class Baseline(LightningModule):
    def __init__(self) -> None:
        super(Baseline, self).__init__()

        self.trainloader, self.dataset, self.train_sampler = build_trainloader()

        # Build model
        # self.app_model, self.app_classifier, self.tsm, self.shape1_classifier,\
        #      self.shape2_classifier, self.fusion_net, self.id_classifier = \
        self.app_model, self.tsm, self.shape_agg, self.shape_classifiers, self.fusion_net, self.id_classifier = \
            build_models(CONFIG, self.dataset.num_pids)
        # Build losses
        self.criterion_cla, self.criterion_pair, self.criterion_shape_mse \
             = build_losses(CONFIG)

        self.training_step_outputs = []
        self.save_hyperparameters()
        
    def configure_optimizers(self):
    
        optimizer = optim.Adam(
            params=self.parameters(),
            lr=CONFIG.TRAIN.OPTIMIZER.LR,
            weight_decay=CONFIG.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=CONFIG.TRAIN.LR_SCHEDULER.STEPSIZE,
            gamma=CONFIG.TRAIN.LR_SCHEDULER.DECAY_RATE)
        
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return self.trainloader
    
    def on_train_epoch_start(self) -> None:
        self.train_sampler.set_epoch(self.current_epoch)
    
    def app_forward(self, clip):
        videowise_app, framewise_app_features = self.app_model(clip)
        return videowise_app, framewise_app_features
    
    def shape_forward(self, input):
        betas, shape_1024s = [], []

        for i in range(input.size(0)):
            beta, shape_1024 = self.tsm(input[i, :, :].unsqueeze(0))
            betas.append(beta)
            shape_1024s.append(shape_1024)
        betas = torch.stack(betas, dim=0)
        shape_1024s = torch.stack(shape_1024s, dim=0)
        framewise_shapes, videowise_shapes = self.shape_agg(shape_1024s) # 16x8x10
        framewise_shape_logits = self.optimize_shape_id_loss(framewise_shapes)
        return betas, framewise_shapes, framewise_shape_logits, videowise_shapes
    
    def optimize_shape_id_loss(self, framewise_shapes): # get framewise shape logits and feed into ID loss
        batch_size, seq_len, _ = framewise_shapes.shape
        batch_framewise_logits_list = []
        for batch_idx in range(batch_size):
            framewise_logits_list = []
            for frame_idx in range(seq_len):
                frame_feature = framewise_shapes[batch_idx][frame_idx]
                logits = self.shape_classifiers[frame_idx](frame_feature)
                # Store the logits for the current frame
                framewise_logits_list.append(logits)
            batch_framewise_logits_list.append(framewise_logits_list)
        final = torch.stack([torch.stack(batch_framewise_logits_list[i]) for i in range(len(batch_framewise_logits_list))])
        return final

    def fusion(self, app_feature, shape_feature):
        final_feature = self.fusion_net(app_feature, shape_feature)
        return final_feature
    
    def training_step(self, batch, batch_idx):
        # clip, pids, _, _, xcs, betas = batch 
        clip, pids, _, _ = batch 
        
        videowise_app, framewise_app_features = self.app_forward(clip)
        
        betas, framewise_shapes, framewise_shape_logits, videowise_shapes = self.shape_forward(framewise_app_features)
             

        fused_feature = self.fusion(videowise_app, videowise_shapes)
        fused_logits = self.id_classifier(fused_feature)

        loss = compute_loss(CONFIG, pids, self.criterion_cla, self.criterion_pair,
                            self.criterion_shape_mse,
                            betas, framewise_shapes, framewise_shape_logits,
                            fused_feature, fused_logits)
        
        acc = FM.accuracy(fused_logits, pids, 'multiclass', average='macro', num_classes=self.dataset.num_pids)
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
        super(Inference, self).__init__()
        
        self.app_model = build_models(config, train=False)
    
    def forward(self, clip):
        features = self.app_model(clip)
        return features

