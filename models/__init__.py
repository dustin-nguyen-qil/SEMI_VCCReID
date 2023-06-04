import logging
from typing import Tuple, Type, Union

from torch import nn

from models.classifier import (Classifier, NormalizedClassifier)
from models.vid_resnet import C2DResNet50
# from models.modules.shape.dsa import DSA
# from models.modules.fusion import FusionNet

def build_models(config, num_ids, num_clothes):
    appearance_model = C2DResNet50(config)
    # shape_model = DSA()
    # fusion = FusionNet(out_features=config.MODEL.FEATURE_DIM)
    shape_model = None
    fusion = None

    if config.TRAIN.WITH_SHAPE:
        identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM,
                                                num_classes=num_ids)
        clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM,
                                                    num_classes=num_clothes)
    else:
        identity_classifier = Classifier(feature_dim=2048, num_classes=num_ids)
            
    return appearance_model, shape_model, fusion, identity_classifier, clothes_classifier
