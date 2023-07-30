from dataclasses import dataclass

@dataclass
class CONFIG:
    @dataclass
    class METADATA:
        LOG_PATH = 'work_space'
        SAVE_PATH = 'work_space/save'

    @dataclass
    class DATA:
        ROOT = 'data'
        DATASET = 'vccr' # vccr, ccvid
        TRAIN_BATCH = 16
        SAMPLING_STEP = 64
        NUM_WORKERS = 4
        HEIGHT = 256
        WIDTH = 128
        TEST_BATCH = 128
        NUM_INSTANCES = 4

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

        RES4_STRIDE = 1
        APP_MODEL = 'c2dres50' # c2d, ap3d, i3d, ap3dnl, nl
        APP_FEATURE_DIM = 2048
        FINAL_FEATURE_DIM = 512
        AGG = 'CONCAT'
        if AGG == 'SUM':
            AGG_FEATURE_DIM = FINAL_FEATURE_DIM
        else:
            AGG_FEATURE_DIM = 2048 + 10

    @dataclass
    class SA: # shape aggregation method
        TYPE = 'asa'
        NUM_FRAME = 8
        NUM_SHAPE_PARAMETERS = 10
        
        @dataclass
        class ASA:
            HIDDEN_SIZE = 1024
            NUM_LAYERS = 2
            FEATURE_POOL = 'attention'
            ATT_LAYERS = 3
            ATT_SIZE = 1024
            ATT_DROPOUT = 0.2

    @dataclass
    class LOSS:
        CLA_LOSS = 'crossentropy'
        CLA_S = 16.
        CLA_M = 0.
        PAIR_LOSS = 'triplet'
        PAIR_S = 16.
        PAIR_M = 0.3
        EPSILON = 0.1
        MOMENTUM = 0.

        SHAPE_LOSS_WEIGHT = 0.5
        FUSED_LOSS_WEIGHT = 1

    @dataclass
    class TRAIN:

        @dataclass
        class LR_SCHEDULER:
            STEPSIZE = 40
            DECAY_RATE = 0.1

        @dataclass
        class OPTIMIZER:
            NAME = 'adam'
            LR = 0.0005
            WEIGHT_DECAY = 5e-4

        START_EPOCH = 0
        MAX_EPOCH = 120
        RESUME = None # add checkpoint here