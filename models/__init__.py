import logging
from typing import Tuple, Type, Union

from torch import nn

from models.classifier import (Classifier, NormalizedClassifier)
from models.vid_resnet import C2DResNet50
from models.dsa import DSA
from models.gaitset import GaitSet
from models.fusion import FusionNet

def build_models(config, num_ids, num_clothes):
    app_model = C2DResNet50(config)
    # shape_model = DSA()
    # fusion = FusionNet(out_features=config.MODEL.FEATURE_DIM)
    shape_model = DSA(num_frames=config.DSA.NUM_FRAME, 
                      num_shape_parameters=config.DSA.NUM_SHAPE_PARAMETERS)
    fusion = FusionNet(out_features=config.MODEL.FINAL_FEATURE_DIM)
    gait_model = GaitSet(config.GAITSET)

    shape1_classifier = Classifier(feature_dim=config.DSA.NUM_SHAPE_PARAMETERS, 
                                        num_classes=num_ids)
    shape2_classifier = Classifier(feature_dim=config.DSA.NUM_SHAPE_PARAMETERS, 
                                        num_classes=num_ids)
    app_classifier = Classifier(feature_dim=config.MODEL.APP_FEATURE_DIM, 
                                        num_classes=num_ids)
    id_classifier = Classifier(feature_dim=config.MODEL.AGG_FEATURE_DIM,
                                                num_classes=num_ids)
        # clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM,
        #                                             num_classes=num_clothes
        # clothes_classifier = None

    return app_model, app_classifier, shape_model, shape1_classifier, shape2_classifier, \
           fusion, id_classifier#, clothes_classifier


def compute_loss(config,
                 pids,
                 criterion_cla, 
                 criterion_pair, 
                 shape_mse_loss,
                 app_feature, 
                 app_logits, 
                 betas,
                 shape1_out,
                 shape1_logits,
                 shape2_feature,
                 shape2_logits,
                 fused_feature, 
                 fused_logits):
    
    shape1_mse = shape_mse_loss(shape1_out, betas)
    shape1_id_loss = criterion_cla(shape1_logits, pids)

    shape2_id_loss = criterion_cla(shape2_logits, pids)
    shape2_pair_loss = criterion_pair(shape2_feature, pids)

    app_id_loss = criterion_cla(app_logits, pids)
    app_pair_loss = criterion_pair(app_feature, pids)

    fused_id_loss = criterion_cla(fused_logits, pids)
    fused_pair_loss = criterion_pair(fused_feature, pids)

    loss = config.LOSS.FUSED_LOSS_WEIGHT * (fused_id_loss + fused_pair_loss) + \
            config.LOSS.APP_LOSS_WEIGHT * (app_id_loss + app_pair_loss) + \
            config.LOSS.SHAPE2_LOSS_WEIGHT * (shape2_id_loss + shape2_pair_loss) + \
            config.LOSS.SHAPE1_LOSS_WEIGHT * (0.1*shape1_id_loss + 0.5*shape1_mse)
    
    return loss 
            
        