import torch
from models.classifier import Classifier
from models.texture.vid_resnet import *
from models.tsm.tsm import TSM
from models.fusion import FusionNet
from models.tsm.dsa import DSA 

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
        tsm = TSM(n_layers=2, hidden_size=1024, add_linear=True, use_residual=True)
        tsm = load_pretrained(tsm, 'work_space/tsm/tsm_model_wo.pth.tar')

        # frame-wise shape aggregation
        shape_agg = DSA(num_frames=config.SA.NUM_FRAME, 
                        num_shape_parameters=config.SA.NUM_SHAPE_PARAMETERS)
        
        fusion = FusionNet(out_features=config.MODEL.FINAL_FEATURE_DIM)

        shape_classifier = Classifier(feature_dim=config.SA.NUM_SHAPE_PARAMETERS, 
                                            num_classes=num_ids)
        # shape2_classifier = Classifier(feature_dim=config.DSA.NUM_SHAPE_PARAMETERS, 
        #                                     num_classes=num_ids)
        # app_classifier = Classifier(feature_dim=config.MODEL.APP_FEATURE_DIM, 
        #                                     num_classes=num_ids)
        id_classifier = Classifier(feature_dim=config.MODEL.AGG_FEATURE_DIM,
                                                    num_classes=num_ids)

        return app_model, tsm, shape_agg, shape_classifier, fusion, id_classifier
        # return app_model, app_classifier, tsm, shape1_classifier, shape2_classifier, \
        #     fusion, id_classifier
    else:
        return app_model
    
def load_pretrained(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    return model


            
        