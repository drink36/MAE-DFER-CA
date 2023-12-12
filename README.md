# MAE-DFER: Efficient Masked Autoencoder for Self-supervised Dynamic Facial Expression Recognition (ACM MM 2023)

> University of Chinese Academy of Sciences & Institute of Automation, Chinese Academy of Sciences & Tsinghua University<br>

## 📰 News


## ✨ Overview

Dynamic Facial Expression Recognition (DFER) is facing **supervised dillema**. On the one hand, current efforts in DFER focus on developing *various* deep supervised models, but only achieving *incremental* progress which is mainly attributed to the *longstanding lack* of large-scale high-quality datasets. On the other hand, due to the *ambiguity* and *subjectivity* in facial expression perception, acquiring large-scale high-quality DFER samples is pretty *time-consuming* and *labor-intensive*. Considering that there are massive unlabeled facial videos on the Internet, this work aims to **explore a new way** (i.e., self-supervised learning) which can fully exploit large-scale *unlabeled* data to largely advance the development of DFER.

<p align="center">
  <img src="figs/Overview.png" width=50%> <br>
  Overview of our MAE-DFER+CA_Module.
</p>

Inspired by recent success of VideoMAE, MAE-DFER makes an early attempt to devise a novel masked autoencoder based self-supervised framework for DFER. It improves VideoMAE by developing an *efficient* LGI-Former as the encoder and introducing *joint* masked appearance and motion modeling. With these two core designs, MAE-DFER *largely* reduces the computational cost (about 38% FLOPs) during fine-tuning while having comparable or even *better* performance.

<p align="center">
  <img src="figs/LGI-Former.png" width=100%> <br>
  The architecture of LGI-Former.
</p>
<!-- ![LGI-Former](figs/LGI-Former.png) -->

Extensive experiments on *six* DFER datasets show that our MAE-DFER *consistently* outperforms the previous best supervised methods by *significant* margins (**\+5∼8%** UAR on three *in-the-wild* datasets and **\+7∼12%** WAR on three *lab-controlled* datasets), which demonstrates that it can learn *powerful* dynamic facial representations for DFER via large-scale self-supervised pre-training. We believe MAE-DFER **has paved a new way** for the advancement of DFER and can inspire more relevant research in this field and even other related tasks (e.g., dynamic micro-expression recognition and facial action unit detection).

## 🚀 Main Results

### ✨ FERV39k

![Result_on_FERV39k](figs/Result_on_FERV39k.png)



## 🔨 Installation

Main prerequisites:

* `Python 3.8`
* `PyTorch 1.7.1 (cuda 10.2)`
* `timm==0.4.12`
* `einops==0.6.1`
* `decord==0.6.0`
* `scikit-learn=1.1.3`
* `scipy=1.10.1`
* `pandas==1.5.3`
* `numpy=1.23.4`
* `opencv-python=4.7.0.72`
* `tensorboardX=2.6.1`

If some are missing, please refer to [environment.yml](environment.yml) for more details.


## ➡️ Data Preparation

Please follow the files (e.g., [ferv39k.py](preprocess/ferv39k.py)) in [preprocess](preprocess) for data preparation.

Specifically, you need to enerate annotations for dataloader ("<path_to_video> <video_class>" in annotations). 
The annotation usually includes `train.csv`, `val.csv` and `test.csv`. The format of `*.csv` file is like:

```
dataset_root/video_1  label_1
dataset_root/video_2  label_2
dataset_root/video_3  label_3
...
dataset_root/video_N  label_N
```

An example of [train.csv](saved/data/ferv39k/all_scenes/train.csv) is shown as follows:

```
/home/drink36/Desktop/Dataset/39K/Face/Action/Happy/0267 0
/home/drink36/Desktop/Dataset/39K/Face/Action/Happy/0316 0
/home/drink36/Desktop/Dataset/39K/Face/Action/Happy/0090 0
```

## 📍Pre-trained Model

Download the model pre-trained on VoxCeleb2 from [this link](https://drive.google.com/file/d/1nzvMITUHic9fKwjQ7XLcnaXYViWTawRv/view?usp=sharing) and put it into [this folder](saved/model/pretraining/voxceleb2/videomae_pretrain_base_dim512_local_global_attn_depth16_region_size2510_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_with_diff_target_server170).

## ⤴️ Fine-tuning with pre-trained models

- FERV39K

    ```
    bash scripts/ferv39k/finetune_local_global_attn_depth16_region_size2510_with_diff_target_164.sh
    ```
  
    The fine-tuning checkpoints and logs for the four different methods of combining CA_Module and MAE-DFER on FERV39K are as follows:
    | Method   | UAR        | WR       |      Fine-tuned   Model            |
    | :------: | :--------: | :------: | :-----------------------:          |
    | add      | 42.30      | 52.40    | [log](saved/model/finetuning/ferv39k/add/nohup.out) / [checkpoint](https://drive.google.com/drive/folders/1DLdfTPgA321QE9rwFNBBGNilN6GKwKrA?usp=drive_link) | 
    | add(4*5) | 41.00      | 51.33    | [log](saved/model/finetuning/ferv39k/add(4*5*10)/nohup.out) / [checkpoint](https://drive.google.com/drive/folders/1lspg-3_DpYoJFqWUg3ZIzhUZUJ6c_SU5?usp=drive_link) | 
    | add(pos) | 42.02      | 52.08    | [log](saved/model/finetuning/ferv39k/add(pos)/nohup.out) / [checkpoint](https://drive.google.com/drive/folders/1EWtMHXcJQ-0PsBeDoXMHueceaJLB4aEb?usp=drive_link) | 
    | cat      | 42.20      | 52.30    | [log](saved/model/finetuning/ferv39k/cat/nohup.out) / [checkpoint](https://drive.google.com/drive/folders/1DLdfTPgA321QE9rwFNBBGNilN6GKwKrA?usp=drive_link) |

## ☎️ Contact 

If you have any questions, please feel free to reach me out at `ooo910809@gmail.com`.

## 👍 Acknowledgements

This project is built upon [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [MAE-DFER](https://github.com/sunlicai/MAE-DFER). Thanks for their great codebase.

