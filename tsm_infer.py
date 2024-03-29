import torch 
import os.path as osp
import pickle
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import transforms as T

from models.tsm.tsm import TSM
from datasets.dataset import InferDataset
from config import CONFIG

device = torch.device('cuda')
model = TSM(n_layers=2, hidden_size=1024, add_linear=True, use_residual=True).to(device)


ckpt = torch.load('work_space/tsm/tsm_model_wo.pth.tar')
ckpt = ckpt['gen_state_dict']
model.load_state_dict(ckpt, strict=False)

# infer training set of the two datasets

vccr_train_set = osp.join(CONFIG.DATA.ROOT, 'vccr', 'train.pkl')
vccr_train_set_w_shape = osp.join(CONFIG.DATA.ROOT, 'vccr', 'train_w_shape.pkl')
ccvid_train_set = osp.join(CONFIG.DATA.ROOT, 'ccvid', 'train.pkl')
ccvid_train_set_w_shape = osp.join(CONFIG.DATA.ROOT, 'ccvid', 'train_w_shape.pkl')

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vccr = InferDataset(vccr_train_set, transform=transform)
ccvid = InferDataset(ccvid_train_set, transform=transform)

datasets = [(vccr, vccr_train_set, vccr_train_set_w_shape), (ccvid, ccvid_train_set, ccvid_train_set_w_shape)]

with torch.inference_mode():
    for i, (dataset, dataset_path, dataset_path_w_shape) in enumerate(datasets):
        with open(dataset_path, 'rb') as f:
            content = pickle.load(f)
        outputs = []
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for img_paths, id, cam_id, clothes_id in tqdm(loader):

            img_paths = img_paths.to(device)
            output = model(img_paths)[-1]
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
        with open(dataset_path_w_shape, 'wb') as f:
            pickle.dump(content, f)






