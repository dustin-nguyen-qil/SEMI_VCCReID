import torch 
import os.path as osp
import pickle
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tsm import TSM
from datasets.dataset import TestDataset
from config import CONFIG

device = torch.device('cuda')
model = TSM(16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True).to(device)


ckpt = torch.load('work_space/tsm/tsm_model_w.pth.tar')#, map_location='cpu')
ckpt = ckpt['gen_state_dict']
model.load_state_dict(ckpt, strict=False)

# infer training set of the two datasets

vccr_train_set = osp.join(CONFIG.DATA.ROOT, 'vccr', 'train.pkl')
ccvid_train_set = osp.join(CONFIG.DATA.ROOT, 'ccvid', 'train.pkl')

vccr = TestDataset(vccr_train_set)
ccvid = TestDataset(ccvid_train_set)

datasets = [(vccr, vccr_train_set), (ccvid, ccvid_train_set)]

with torch.inference_mode():
    for i, (dataset, dataset_path) in enumerate(datasets):
        with open(dataset_path, 'rb') as f:
            content = pickle.load(f)
        outputs = []
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for clip, id, cam_id, clothes_id, img_paths in tqdm(loader):

            clip = clip.to(device)
            output = model(clip)[-1]
            seqlen = len(img_paths)
            pred_shape_1024 = (output['shape_1024'].reshape(seqlen, -1)).detach().cpu().numpy()
            pred_betas = output['theta'][:, :, 75:].reshape(seqlen, -1).detach().cpu().numpy()

            outputs.append({
                'p_id': int(id.numpy()[0]),
                'cam_id': int(cam_id.numpy()[0]),
                'clothes_id': int(clothes_id.numpy()[0]),
                'img_paths': img_paths,
                'shape_1024': pred_shape_1024,
                'betas': pred_betas.tolist(),
            })
        content['data'] = output
        with open(dataset_path, 'wb') as f:
            pickle.dump(content, f)






