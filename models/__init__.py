import logging
from typing import Tuple, Type, Union

from torch import nn

from models.classifier import Classifier
from models.texture.vid_resnet import *
from models.tsm.dsa import DSA
from models.fusion import FusionNet

__factory = {
    'c2dres50': C2DResNet50,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
}

def build_models(config, num_ids: int = 150, train=True):


    if config.MODEL.APP_MODEL not in __factory.keys():
        raise KeyError("Invalid model: '{}'".format(config.MODEL.APP_MODEL))
    else:
        app_model = __factory[config.MODEL.APP_MODEL](config)

    if train:
        shape_model = DSA(num_frames=config.DSA.NUM_FRAME, 
                        num_shape_parameters=config.DSA.NUM_SHAPE_PARAMETERS)
        fusion = FusionNet(out_features=config.MODEL.FINAL_FEATURE_DIM)

        shape1_classifier = Classifier(feature_dim=config.DSA.NUM_SHAPE_PARAMETERS, 
                                            num_classes=num_ids)
        shape2_classifier = Classifier(feature_dim=config.DSA.NUM_SHAPE_PARAMETERS, 
                                            num_classes=num_ids)
        app_classifier = Classifier(feature_dim=config.MODEL.APP_FEATURE_DIM, 
                                            num_classes=num_ids)
        id_classifier = Classifier(feature_dim=config.MODEL.AGG_FEATURE_DIM,
                                                    num_classes=num_ids)

        return app_model, app_classifier, shape_model, shape1_classifier, shape2_classifier, \
            fusion, id_classifier
    else:
        return app_model


            
        