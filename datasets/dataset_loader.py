from torch.utils.data import DataLoader
from datasets.dataset import VideoDataset, TestDataset
from config import CONFIG as config
import os.path as osp
import datasets.spatial_transforms as ST
import datasets.temporal_transforms as TT 
from datasets.samplers import RandomIdentitySampler

def build_transforms():
    spatial_transform_train = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ST.RandomErasing(height=config.DATA.HEIGHT,
                         width=config.DATA.WIDTH,
                         probability=config.AUG.RE_PROB)
    ])

    spatial_transform_test = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
        temporal_transform_train = TT.TemporalDivisionCrop(
            size=config.AUG.SEQ_LEN)
    elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
        temporal_transform_train = TT.TemporalRandomCrop(
            size=config.AUG.SEQ_LEN, stride=config.AUG.SAMPLING_STRIDE)
    else:
        raise KeyError("Invalid temporal sempling mode '{}'".format(
            config.AUG.TEMPORAL_SAMPLING_MODE))

    temporal_transform_test = None

    return spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test

st_train, st_test, tt_train, tt_test = build_transforms()

def build_trainloader():

    """
    Build Train Loader
    """
    train_data_path = osp.join(config.DATA.ROOT, config.DATA.DATASET, 'train.pkl')

    train = VideoDataset(
        train_data_path, 
        st_train, 
        tt_train, 
        dense_sampling=config.DATA.TRAIN_DENSE
        )
    
    if config.DATA.USE_SAMPLER:
        sampler = RandomIdentitySampler(train.dataset, config.DATA.NUM_INSTANCES)

        trainloader = DataLoader(
            train, 
            batch_size=config.DATA.TRAIN_BATCH,
            sampler=sampler,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, 
            drop_last=True
        )
    else:
        trainloader = DataLoader(
            train, 
            shuffle=True,
            batch_size=config.DATA.TRAIN_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, 
            drop_last=True
        )
    if config.DATA.USE_SAMPLER:
        return trainloader, train, sampler
    else:
        return trainloader, train

def build_testloader():
    """
    Build query and gallery loader
    """
    query_data_path = osp.join(config.DATA.ROOT, config.DATA.DATASET, 'query.pkl')
    gallery_data_path = osp.join(config.DATA.ROOT, config.DATA.DATASET, 'gallery.pkl')

    query = TestDataset(
        query_data_path,
        spatial_transform=st_test,
        temporal_transform=tt_test,
        seq_len=config.AUG.SEQ_LEN,
        stride = config.AUG.SAMPLING_STRIDE
    )

    gallery = TestDataset( 
        gallery_data_path,
        spatial_transform=st_test,
        temporal_transform=tt_test,
        seq_len=config.AUG.SEQ_LEN,
        stride = config.AUG.SAMPLING_STRIDE
    )

    queryloader = DataLoader(
        query, 
        batch_size=config.DATA.TEST_BATCH, 
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, 
        drop_last=False
    )
    galleryloader = DataLoader(
        gallery, 
        batch_size=config.DATA.TEST_BATCH, 
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, 
        drop_last=False
    )

    return queryloader, galleryloader, query, gallery