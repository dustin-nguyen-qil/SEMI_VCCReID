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
        DATASET = 'vccr' # vccr, ccvid, ccpg
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
        APP_MODEL = 'c2dres50' # c2dres50, ap3dres50, i3dres50, ap3dnlres50, nlres50
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

        SHAPE_LOSS_WEIGHT = 0.1
        FUSED_LOSS_WEIGHT = 1

    @dataclass
    class TRAIN:

        @dataclass
        class LR_SCHEDULER:
            STEPSIZE = 20
            DECAY_RATE = 0.1

        @dataclass
        class OPTIMIZER:
            NAME = 'adam'
            LR = 0.0003
            WEIGHT_DECAY = 5e-4
        """
            pose: 'front' #standard #wo_face # back # size # front_size # front_back # side_back 
            cloth: one_cloth, two_cloth, three cloth, up, down, al
        """
        TYPE = 'pose' # pose, cloth
        TRAIN_MODE = 'standard' 
        START_EPOCH = 0
        MAX_EPOCH = 60
        RESUME = None # add checkpoint here
    
    @dataclass
    class TEST:
        TEST_MODE = 'all'
        TRAIN_SET = 'vccr'
        TEST_SET = 'vccr'
        
        
