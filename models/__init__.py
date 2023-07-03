import logging
from typing import Tuple, Type, Union

from torch import nn

from models.classifier import (Classifier, NormalizedClassifier)
from models.appearance.vid_resnet import *
from models.appearance.TCLNet import TCLNet
from models.appearance.BiCnet_TKS import BiCnet_TKS
from models.dsa import DSA
from models.gaitset import GaitSet
from models.fusion import FusionNet

def build_models(config, num_ids, num_clothes, train=True):

    if config.MODEL.APP_MODEL == 'c2d':
        app_model = C2DResNet50(config)
    elif config.MODEL.APP_MODEL == 'ap3d':
        app_model = AP3DResNet50(config)
    elif config.MODEL.APP_MODEL == 'ap3dnl':
        app_model = AP3DNLResNet50(config)
    elif config.MODEL.APP_MODEL == 'i3d':
        app_model = I3DResNet50(config)
    elif config.MODEL.APP_MODEL == 'tclnet':
        app_model = TCLNet(num_classes=num_ids, use_gpu=True)
    elif config.MODEL.APP_MODEL == 'bicnet':
        app_model = BiCnet_TKS()

    if train:
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
    else:
        return app_model


            
        