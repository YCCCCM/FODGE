<div align="center">
<h2><font color="red"> 👠👠👠 FODGE 👠👠👠 </font></center> <br> <center>HIGH-FIDELITY DANCE GENERATION VIA FULL-BODY OPTIMIZATION</h2>

Xiaoying Huang, Sanyi Zhang, Qin Zhang, Xiaoxuan Guo and Long Ye

<a href='https://yccccm.github.io/FODGE-page/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
<a href=''><img src='https://img.shields.io/badge/ArXiv-0000.00000-red'></a> 
</div>

Code organizing in progress！！！

## Abstract
Dance is an important connection of artistic expression, yet automatically creating new choreography especially music-driven high-fidelity dance generation still faces significant challenges. Existing approaches primarily enhance physical realism by optimizing foot–ground contact, but some local parts are still unsatisfactory which reduce expressiveness, such as overlook gliding steps (e.g., moonwalk), unnatural arm and head movements. To address these issues, we propose FODGE, a music-conditioned diffusion-based framework with full-body optimization. FODGE integrates a Full-body Refinement Block (FRB) and a Full-body Optimization Post-processing module (FOP). FRB introduces and learns the relationship between arm and foot movements as an optimization clue to refine human motions, thereby enhancing artistic expressiveness. And we further employ a training-free FOP to optimize the dance sequence from a full-body perspective, improving visualization effects. Experimental results demonstrate that FODGE significantly outperforms existing baselines in motion quality while achieving a better balance between physical plausibility and artistic expressiveness..</b>

## Setup Environment
Our method is trained using cuda12.1, pytorch-lightning 1.9.5 on 4 Nvidia L40 GPUs.
``` 
conda env create -f dancedm.yaml
or
pip install requirements_5090_cu128.txt
```
* We recommend Linux for performance and compatibility reasons. Windows is OK, please see `dancedm_win.yaml`.
* 64-bit Python 3.10
* PyTorch 2.3.1
* At least 48 GB RAM per GPU
* 4+ high-end NVIDIA GPUs with at least 48 GB of GPU memory, NVIDIA drivers, CUDA 12.1 toolkit.

The train and inference example build this repo was validated on:
* Ubuntu 24.04 LTS
* 64-bit Python 3.10
* PyTorch 2.9.0 or PyTorch 2.3.1
* 96 GB RAM or 256 GB RAM
* 1 x NVIDIA Geforce RTX5090, CUDA 12.8 toolkit or 8 x NVIDIA A800/L40, CUDA 12.4 toolkit



### Data preparation

Please visit [FineDance](https://github.com/li-ronghui/FineDance) and [LODGE](https://github.com/li-ronghui/LODGE) to download the origin FineDance dataset and put it in the ./data floder. 

The file structure is as follows:

```bash
DanceDM
├── data
│   ├── code
│   │   ├──preprocess.py
│   │   ├──extract_musicfea35.py
│   ├── finedance
│   │   ├──label_json
│   │   ├──motion
│   │   ├──music_npy
│   │   ├──music_wav
│   │   ├──music_npynew
│   │   ├──mofea319
│   │── Normalizer.pth
└   └── smplx_neu_J_1.npy
```

## Model Training

Traing the Local Diffusion and Global Diffusion
```bash
python train_teacher.py --cfg configs/dancedm/local/teacher/local_fea139_teacher.yaml --cfg_assets configs/data/assets.yaml 
python train_teacher.py --cfg configs/dancedm/global/teacher/global_fea139_teacher.yaml --cfg_assets configs/data/assets.yaml
```

Set the pretrained Local Diffusion checkpoint path at the "TRAIN.PRETRAINED" of "configs/dancedm/local/body_opt/local_fea139_teacher_body_opt.yaml", then finetuning the Local Diffusion for smooth generation.
```bash
python train_body_opt.py --cfg configs/dancedm/local/body_opt/local_fea139_teacher_body_opt.yaml --cfg_assets configs/data/assets.yaml
```

## Inference
Once the training is done, run inference:
```bash
python -m inference.infer_teacher_  --cfg experiments/Local_Module/debug--Local_Module_teacher/config_xxxx-xx-xx-xx-xx-xx_train.yaml --soft 1.0
```
After the inference is completed, please run the post-processing module
```bash
python whole_body_process.py
```

## Blender 3D rendering
In order to render generated dances in 3D, we convert them into FBX files to be used in Blender. We provide a sample rig, `SMPL-to-FBX/ybot.fbx`.
After generating dances with the `--save-motions` flag enabled, move the relevant saved `.pkl` files to a folder, e.g. `smpl_samples`
Run
```.bash
python SMPL-to-FBX/Convert.py --input_dir SMPL-to-FBX/smpl_samples/ --output_dir SMPL-to-FBX/fbx_out
```
to convert motions into FBX files, which can be imported into Blender and retargeted onto different rigs, i.e. from [Mixamo](https://www.mixamo.com). A variety of retargeting tools are available, such as the [Rokoko plugin for Blender](https://www.rokoko.com/integrations/blender).



## Citation 
If you think this project is helpful, please cite our paper:
```bibtex
@inproceedings{huang2026fodge,
  title={FODGE: HIGH-FIDELITY DANCE GENERATION VIA FULL-BODY OPTIMIZATION},
  author={Xiaoying Huang, Sanyi Zhang, Qin Zhang, Xiaoxuan Guo and Long Ye},
  booktitle={underview},
  year={2026},
}
``` 


## Acknowledgements

This basic dance diffusion borrows from [EDGE](https://github.com/Stanford-TML/EDGE), [FineDance](https://github.com/li-ronghui/FineDance) and [LODGE](https://github.com/li-ronghui/LODGE).
