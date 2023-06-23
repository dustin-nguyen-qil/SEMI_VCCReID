import logging
from typing import Tuple, Type, Union

from torch import nn

from models.classifier import (Classifier, NormalizedClassifier)
from models.vid_resnet import C2DResNet50
from models.dsa import DSA
from models.gaitset import GaitSet
from models.fusion import FusionNet

def build_models(config, num_ids, num_clothes):
    appearance_model = C2DResNet50(config)
    # shape_model = DSA()
    # fusion = FusionNet(out_features=config.MODEL.FEATURE_DIM)
    shape_model = DSA(num_frames=config.DSA.NUM_FRAME, 
                      num_shape_parameters=config.DSA.NUM_SHAPE_PARAMETERS)
    fusion = FusionNet(out_features=config.MODEL.FINAL_FEATURE_DIM)
    gait_model = GaitSet(config.GAITSET)

    if config.TRAIN.WITH_SHAPE:
        identity_classifier = Classifier(feature_dim=config.MODEL.AGG_FEATURE_DIM,
                                                num_classes=num_ids)
        # clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM,
        #                                             num_classes=num_clothes)
    else:
        identity_classifier = Classifier(feature_dim=config.MODEL.APP_FEATURE_DIM, num_classes=num_ids)
        # clothes_classifier = None

    return appearance_model, shape_model, fusion, identity_classifier#, clothes_classifier
