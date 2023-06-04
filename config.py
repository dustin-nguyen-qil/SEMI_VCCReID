from dataclasses import dataclass
import os.path as osp
import torch 
from torchvision import transforms as T

@dataclass
class CONFIG:
    @dataclass
    class METADATA:
        OUTPUT = 'work_space/logs'
        MODEL_NAME = '...'

    @dataclass
    class DATA:
        ROOT = 'data'
        DATASET = 'ccvid' # vccr
        USE_SAMPLER = False
        SAMPLING_STEP = 64
        NUM_WORKERS = 4
        HEIGHT = 256
        WIDTH = 128
        TRAIN_BATCH = 32
        TEST_BATCH = 128
        NUM_INSTANCES = 4
        TRAIN_DENSE = False

    @dataclass
    class AUG:
        RE_PROB = 0.0
        TEMPORAL_SAMPLING_MODE = 'stride'
        SEQ_LEN = 8
        SAMPLING_STRIDE = 4

    @dataclass
    class MODEL:

        @dataclass
        class AP3D:
            TEMPERATURE = 4
            CONTRACTIVE_ATT = True

        NAME = 'c2dres50'
        RES4_STRIDE = 1
        FEATURE_DIM = 512
        AGG = 'SUM'
        RESUME = ''  # add checkpoint path here

    @dataclass
    class LOSS:
        CLA_LOSS = 'crossentropy'
        CLA_S = 16.
        CLA_M = 0.
        CLOTHES_CLA_LOSS = 'cosface'
        PAIR_LOSS = 'triplet'
        PAIR_LOSS_WEIGHT = 0.0
        PAIR_S = 16.
        PAIR_M = 0.3
        CAL = 'cal'
        EPSILON = 0.1
        MOMENTUM = 0.

    @dataclass
    class TRAIN:

        @dataclass
        class LR_SCHEDULER:
            STEPSIZE = [40, 80, 120]
            DECAY_RATE = 0.1

        @dataclass
        class OPTIMIZER:
            NAME = 'adam'
            LR = 0.00035
            WEIGHT_DECAY = 5e-4

        WITH_SHAPE = False
        START_EPOCH = 0
        MAX_EPOCH = 150
        # START_EPOCH_CC = 50
        # START_EPOCH_ADV = 50