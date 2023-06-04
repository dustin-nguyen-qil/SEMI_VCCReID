import functools
import os
from typing import Optional
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import math


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print(
                "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill."
                .format(img_path))
            pass
    return img


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_default_image_loader():
    return pil_loader


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

def densesampling_for_trainingset(dataset, sampling_step=64):
    ''' Split all videos in training set into lots of clips for dense sampling.

    Args:
        dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
        sampling_step (int): sampling step for dense sampling

    Returns:
        new_dataset (list): output dataset
    '''
    new_dataset = []
    for (img_paths, pid, camid, clothes_id) in dataset:
        if sampling_step != 0:
            num_sampling = len(img_paths) // sampling_step
            if num_sampling == 0:
                new_dataset.append((img_paths, pid, camid, clothes_id))
            else:
                for idx in range(num_sampling):
                    if idx == num_sampling - 1:
                        new_dataset.append(
                            (img_paths[idx * sampling_step:], pid, camid,
                                clothes_id))
                    else:
                        new_dataset.append(
                            (img_paths[idx * sampling_step:(idx + 1) *
                                        sampling_step], pid, camid,
                                clothes_id))
        else:
            new_dataset.append((img_paths, pid, camid, clothes_id))

    return new_dataset

def recombination_for_testset(dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths) // (seq_len * stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx:end_idx:stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len*stride
            if len(img_paths) % (seq_len * stride) != 0:
                # reducing stride
                new_stride = (len(img_paths) % (seq_len * stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len * stride) * (
                        seq_len * stride) + i
                    end_idx = len(img_paths) // (seq_len * stride) * (
                        seq_len * stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx:end_idx:new_stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths) // seq_len *
                                           seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert ((vid2clip_index[idx, 1] -
                     vid2clip_index[idx, 0]) == math.ceil(
                         len(img_paths) / seq_len))

        return new_dataset, vid2clip_index.tolist()


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
        self.loader = get_loader()
        self.train_mode = train
        if self.train_mode:
            if dense_sampling:
                self.dataset = densesampling_for_trainingset(data)
            else:
                self.dataset = data
        else:
            self.dataset = recombination_for_testset(data, 
                                                     seq_len, stride)

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

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        tracklet = self.dataset[index]
        
        img_paths = tracklet['img_paths']
        pid = tracklet['p_id']
        camid = tracklet['cam_id']
        clothes_id = tracklet['clothes_id']
    
        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)
        
        subset_indices = [img_paths.index(img_path) for img_path in img_paths]

        if self.train_mode:
            xcs = tracklet['shape_1024']
            betas = tracklet['betas']
            xc = [xcs[i] for i in subset_indices]
            beta = [betas[i] for i in subset_indices]

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
        