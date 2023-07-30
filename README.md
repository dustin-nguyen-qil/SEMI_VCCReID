# SEMI: Temporal 3D Shape Modeling for Video-based Cloth-changing Person Re-Identification [WACV 2024]

This repository contains official implementation for the paper: [Temporal 3D Shape Modeling for Video-based Cloth-changing Person Re-Identification](). 

#### Quantitative Results

[//]: <> add table of quantitative results here.

## 1. Features

#### Supported CNN backbones

- `c2dres50`: C2DResNet50
- `i3dres50`: I3DResNet50
- `ap3dres50`: AP3DResNet50
- `nlres50`: NLResNet50
- `ap3dnlres50`: AP3DNLResNet50

#### Summary of VCCRe-ID datasets

This baseline currently supports two public VCCRe-ID datasets: **VCCR** and **CCVID**.

| Dataset | Paper | Num.IDs | Num.Tracklets | Num.Clothes/ID | Public | Download |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Motion-ReID | [link](https://ieeexplore.ieee.org/document/8354164)| 30 | 240 | - | X | - |
| CVID-reID |  [link](https://ieeexplore.ieee.org/document/9214519)| 90 | 2980 | - | X | - |
| SCCVRe-ID |  [link](https://arxiv.org/abs/2211.11165)| 333 | 9620 | 2~37 | X | - |
| RCCVRe-ID |  [link](https://arxiv.org/abs/2211.11165)| 34 | 6948 | 2~10 | X | - |
| CCPG |  [link](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_An_In-Depth_Exploration_of_Person_Re-Identification_and_Gait_Recognition_in_CVPR_2023_paper.pdf)| 200 | ~16k | - | Per Request | [project link](https://github.com/BNU-IVC/CCPG) |
| CCVID |  [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Clothes-Changing_Person_Re-Identification_With_RGB_Modality_Only_CVPR_2022_paper.pdf)| 226 | 2856 | 2~5 | Yes | [link](https://drive.google.com/file/d/1vkZxm5v-aBXa_JEi23MMeW4DgisGtS4W/view?usp=sharing) |
| VCCR |  [link](https://openaccess.thecvf.com/content/ACCV2022/papers/Han_3D_Shape_Temporal_Aggregation_for_Video-Based_Clothing-Change_Person_Re-identification_ACCV_2022_paper.pdf)| 392 | 4384 | 2~10 | Yes | [link](https://drive.google.com/file/d/17qJPksE-Fk189KSHTPYQihMfnzXnHC6m/view) |

## 2. Running instructions

### Getting started

#### Create virtual environment

First, create a virtual environment for the repository
```bash
conda create -n semi python=3.8
```
then activate the environment 
```bash
conda activate semi
```


#### Clone the repository

```bash
git clone https://github.com/dustin-nguyen-qil/Video-based-Cloth-Changing-ReID-Baseline.git
```
Next, install the dependencies by running
...
```bash
pip install -r requirements.txt
```

### Data Preparation

1. Download the datasets VCCR and CCVID following download links above
2. You need the pickle files containing the paths to sequence images, clothes id, identity and camera id of the sequences. To do this:
    - Create a folder named `data` inside the repository
    - Run the following command line (**Note**: replace the path to the folder storing the datasets and the dataset name)

```bash
python datasets/prepare.py --root "/media/dustin/DATA/Research/Video-based ReID" --dataset_name vccr
```

### Configuration options

Go to `./config.py` to modify configurations if needed: Dataset (VCCR or CCVID), number of epochs, batch size learning rate, CNN backbone (according to model names above), etc.

### Run baseline

Create a folder named `work_space` as below.

Download the pretrained SPIN model and the SMPL mean parameters needed to train the 3D regressor from [here](https://uofh-my.sharepoint.com/:f:/g/personal/dnguy222_cougarnet_uh_edu/EksKjRj1EDpMurG85R79_7kBO95Mu_nFxPuMdMmFSKZkZg?e=1TDLnW) (password: dustinqil). Put it inside `work_space`.

```
data
work_space
|--- save
|--- output
|--- tsm
main.sh
```

#### Run training and testing

```bash
bash main.sh
```

- Checkpoints will be automatically saved to `work_space/lightning_logs`.
- Trained model will be automatically saved to `work_space/save`.
- Testing results will be automatically saved to `work_space/output`.

If you want to train from checkpoint, add checkpoint path to RESUME in `config.py`. 

#### Run evaluation only

- Download our pretrained model from [here](), put it in `work_space/save`.
- Replace the path to the pretrained model in `test.py`
- Run 
```bash
python test.py
```

## Citation

If you find this repo helpful, please cite:

```bash
@inproceedings{vuong2024semi,
  title={Temporal 3D Shape Modeling for Video-based Cloth-changing Person Re-Identification},
  author={Nguyen, Vuong and Shah, Shishir and Mantini, Pranav},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year = {2024}
}
```

## Acknowledgement

Related repos: 
- [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID). 













