from torch import nn
from utils.losses.cross_entropy_loss_with_label_smooth import CrossEntropyWithLabelSmooth
from utils.losses.triplet_loss import TripletLoss
from utils.losses.contrastive_loss import ContrastiveLoss
from utils.losses.arcface_loss import ArcFaceLoss
from utils.losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from utils.losses.circle_loss import CircleLoss, PairwiseCircleLoss

def build_losses(config, num_train_clothes):
    # Build identity classification loss
    if config.LOSS.CLA_LOSS == 'crossentropy':
        criterion_cla = nn.CrossEntropyLoss()
    elif config.LOSS.CLA_LOSS == 'crossentropylabelsmooth':
        criterion_cla = CrossEntropyWithLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'arcface':
        criterion_cla = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'circle':
        criterion_cla = CircleLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))

    # Build pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterion_pair = TripletLoss(margin=config.LOSS.PAIR_M, distance='cosine')
    elif config.LOSS.PAIR_LOSS == 'contrastive':
        criterion_pair = ContrastiveLoss(scale=config.LOSS.PAIR_S)
    elif config.LOSS.PAIR_LOSS == 'cosface':
        criterion_pair = PairwiseCosFaceLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'circle':
        criterion_pair = PairwiseCircleLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    else:
        raise KeyError("Invalid pairwise loss: '{}'".format(config.LOSS.PAIR_LOSS))
    
    # Build losses for shape module
    criterion_shape_mse = nn.MSELoss()
    return criterion_cla, criterion_pair, criterion_shape_mse
    

def compute_loss(config,
                 pids,
                 criterion_cla, 
                 criterion_pair, 
                 criterion_shape_mse,
                 app_feature, 
                 app_logits, 
                 betas,
                 shape1_out,
                 shape1_logits,
                 shape2_feature,
                 shape2_logits,
                 fused_feature, 
                 fused_logits):
    
    shape1_mse = criterion_shape_mse(shape1_out, betas)
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