import pickle
import torch
from torch.utils.data import Dataset

from datasets.utils import *

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 data_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 dense_sampling=False,
                 train=True,
                 seq_len: int=16,
                 stride: int=4
                ):
        data, self.num_pids, self.num_clothes = self.read_dataset(data_path)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader
        self.train_mode = train
        self.train_dense = dense_sampling
        if self.train_mode:
            if self.train_dense:
                self.dataset = densesampling_for_trainingset(data)
            else:
                self.dataset = data
        else:
            self.dataset = recombination_for_testset(data, seq_len, stride)

    def __len__(self):
        return len(self.dataset)
    
    def read_dataset(self, data_path):
        with open(data_path, 'rb') as f:
            content = pickle.load(f)
        data = content['data']
        num_pids = content['num_pids']
        num_clothes = content['num_clothes']
        return data, num_pids, num_clothes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            img_paths = tracklet['img_paths']
            pid = tracklet['p_id']
            camid = tracklet['cam_id']
            clothes_id = tracklet['clothes_id']
            xcs = tracklet['shape_1024']
            betas = tracklet['betas']

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
            
        """
        tracklet = self.dataset[index]
        if self.train_dense:
            (pid, camid, clothes_id, img_paths, xcs, betas, _, _, _) = tracklet
        else:
            (pid, camid, clothes_id, img_paths, xcs, betas, _, _, _) = list(tracklet.values())
    
        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)
        
        subset_indices = [img_paths.index(img_path) for img_path in img_paths]

        if self.train_mode:
            xc = [xcs[i] for i in subset_indices]
            beta = [betas[i] for i in subset_indices]
        
        img_paths = [img_path[0] for img_path in img_paths]
        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if self.train_mode:
            return clip, pid, camid, clothes_id, xc, beta
        else:
            return clip, pid, camid, clothes_id
        