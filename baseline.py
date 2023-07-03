from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.optim import lr_scheduler
from config import CONFIG
from datasets.dataset_loader import build_trainloader
from models import build_models, compute_loss
from utils.losses import build_losses
from utils.multiloss_weighting import MultiNoiseLoss
from torchmetrics import functional as FM
from models.vid_resnet import *

class Baseline(LightningModule):
    def __init__(self) -> None:
        super(Baseline, self).__init__()

        if CONFIG.DATA.USE_SAMPLER:
            self.trainloader, self.dataset, self.train_sampler = build_trainloader()
        else:
            self.trainloader, self.dataset = build_trainloader()
        
        # pid2clothes = torch.from_numpy(self.dataset.pid2clothes)

        # Build model
        self.app_model, self.app_classifier, self.shape_model, self.shape1_classifier,\
             self.shape2_classifier, self.fusion_net, self.id_classifier = \
            build_models(CONFIG, self.dataset.num_pids, self.dataset.num_clothes)
        # Build identity classification loss, pairwise loss, clothes classificaiton loss
        # and adversarial loss.
        self.criterion_cla, self.criterion_pair, self.criterion_shape_mse, _, _ \
             = build_losses(CONFIG, self.dataset.num_clothes)
        if CONFIG.LOSS.MULTI_LOSS_WEIGHTING:
            self.multi_loss = MultiNoiseLoss(n_losses=4)
        else: self.multi_loss = None

        self.training_step_outputs = []
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        if CONFIG.LOSS.MULTI_LOSS_WEIGHTING:
            base_params = list(self.app_model.parameters()) + list(self.app_classifier.parameters()) +\
                            list(self.shape_model.parameters()) + list(self.shape1_classifier.parameters()) +\
                            list(self.shape2_classifier.parameters()) + list(self.fusion_net.parameters()) +\
                            list(self.id_classifier.parameters()) 
            optimizer = optim.Adam(
                [
                    {'params': base_params},
                    {'params': self.multi_loss.noise_params, 'lr': 0.001}
                ],
                lr=CONFIG.TRAIN.OPTIMIZER.LR,
                weight_decay=CONFIG.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        else:
            optimizer = optim.Adam(
                params=self.parameters(),
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
        if CONFIG.DATA.USE_SAMPLER:
            self.train_sampler.set_epoch(self.current_epoch)
        else:
            pass
    
    def app_forward(self, images):
        app_feature = self.app_model(images)
        app_logits = self.app_classifier(app_feature)
        return app_feature, app_logits
    
    def shape_forward(self, xcs):
        shape1_out, shape1_feature, shape2_feature = self.shape_model(xcs)
        shape1_logits = self.shape1_classifier(shape1_feature)
        shape2_logits = self.shape2_classifier(shape2_feature)
        return shape1_out, shape2_feature, shape1_logits, shape2_logits
    
    def fusion(self, app_feature, shape_feature):
        final_feature = self.fusion_net(app_feature, shape_feature)
        return final_feature
    
    def training_step(self, batch, batch_idx):
        imgs, pids, _, clothes_ids, xcs, betas = batch 
        app_feature, app_logits = self.app_forward(imgs)
        shape1_out, shape2_feature, shape1_logits, shape2_logits = \
                self.shape_forward(xcs)
        
        fused_feature = self.fusion(app_feature, shape2_feature)
        fused_logits = self.id_classifier(fused_feature)

        loss = compute_loss(CONFIG, pids, self.criterion_cla, self.criterion_pair,
                            self.criterion_shape_mse, app_feature, app_logits,
                            betas, shape1_out, shape1_logits, shape2_feature, shape2_logits,
                            fused_feature, fused_logits, self.multi_loss)
        
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
        if config.MODEL.APP_MODEL == 'c2d':
            self.app_model = C2DResNet50(config)
        elif config.MODEL.APP_MODEL == 'ap3d':
            self.app_model = AP3DResNet50(config)
        if config.MODEL.APP_MODEL == 'ap3dnl':
            self.app_model = AP3DNLResNet50(config)
        if config.MODEL.APP_MODEL == 'i3d':
            self.app_model = I3DResNet50(config)
    
    def forward(self, imgs):
        return self.app_model(imgs)

