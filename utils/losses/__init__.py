from torch import nn
from utils.losses.cross_entropy_loss_with_label_smooth import CrossEntropyWithLabelSmooth
from utils.losses.triplet_loss import TripletLoss
from utils.losses.contrastive_loss import ContrastiveLoss
from utils.losses.arcface_loss import ArcFaceLoss
from utils.losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from utils.losses.circle_loss import CircleLoss, PairwiseCircleLoss

def build_losses(config):
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
                 betas,
                 framewise_shape, 
                 framewise_shape_logits,
                 fused_feature, 
                 fused_logits):
                
    
    shape_mse = criterion_shape_mse(framewise_shape, betas)
    _, seq_len, num_pids = framewise_shape_logits.shape
    shape_pids = pids.repeat_interleave(seq_len)
    shape_id_loss = criterion_cla(framewise_shape_logits.view(-1, num_pids), shape_pids)

    fused_id_loss = criterion_cla(fused_logits, pids)
    fused_pair_loss = criterion_pair(fused_feature, pids)

    loss = config.LOSS.FUSED_LOSS_WEIGHT * (0.5*fused_id_loss + 0.5*fused_pair_loss) + \
            config.LOSS.SHAPE_LOSS_WEIGHT * (0.1*shape_id_loss + 0.8*shape_mse)
            
    return loss 