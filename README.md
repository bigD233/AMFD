# AMFD: Distillation via Adaptive Multimodal Fusion for Multispectral Pedestrian Detection
[Arxiv](https://export.arxiv.org/abs/2405.12944) | [Kaggle](https://www.kaggle.com/datasets/zizhaochen6/sjtu-multispectral-object-detection-smod-dataset) |

<img src=".\\img\\overall.png" width=95% title="origin input"/>

<div align="left">
<img src=".\\img\\origin_rgb.png" width=23% title="origin input"/><img src=".\\img\\teacher_fused.png" width=25% title="teacher"/><img src=".\\img\\distill_fused.png" width=25% title="student distill fusion feature"/><img src=".\\img\\distill_multi.png" width=25% title="student distill by AMFD"/>
</div>

This is the official repository for our paper "AMFD: Distillation via Adaptive Multimodal Fusion for Multispectral Pedestrian Detection" ([arxiv paper link](https://export.arxiv.org/abs/2405.12944)).


## Preparation

1. We use the [MMDetection](https://github.com/open-mmlab/mmdetection) toolbox to detect pedestrians on the KAIST, LLVIP and SMOD dataset. Please follow the [MMDetection](https://github.com/open-mmlab/mmdetection) documents to install environments.

	In our environment, we use:
	```
	python==3.9.1
	pytorch==1.12.1+cu116+cudnn8_0
	torchvision==0.13.1+cu116
	mmcv==2.0.1
	mmdet==3.1.0
	mmengine==0.8.4
	```
2. Go to the folder where mmdetection is located and clone our project to the projects folder in that directory.
	```
	cd mmdetection
	cd projects
	git clone https://github.com/bigD233/AMFD.git
	```

## Datasets and Models
All Datasets and Models we provided are upload in [google cloud](https://drive.google.com/drive/folders/1iAjAVifSSuizjEDFao8Ba3heZFsBNoae?usp=sharing).
### Datasets
- KAIST: KAIST dataset has been updated by several previous works. We upload this dataset and improved annotations for your convenience in using our code. [cloud link](https://drive.google.com/drive/folders/1lJLGV91CH43PRYx8KaVy5OngryTwo3ee?usp=sharing)
- LLVIP: The origin data you can download in its from its official [repository](https://github.com/bupt-ai-cz/LLVIP). The coco-format annotation you can download from [cloud link](https://drive.google.com/drive/folders/1A_6o6OQuMOZpzp6S8Zzm0jbmn5bh7x6E?usp=sharing).
- SMOD: A new multispectral object detection dataset propsed by us. You can download from [Kaggle](https://www.kaggle.com/datasets/zizhaochen6/sjtu-multispectral-object-detection-smod-dataset)

## Models 
We provide the teacher checkpoint "Teacher_Fasterrcnn_7_66.pth" and the studnet checkpoint "single_fasterrcnn_7_23.pth" trained on KAIST dataset. [cloud link](https://drive.google.com/drive/folders/1UEop1kPMKfRQHiYIW_d6hPdHMHhNBWLn?usp=sharing)
```
Teacher_Fasterrcnn_7_66.pth ---> Trained by Teacher_Fasterrcnn_r50_fpn_1x_kaist_thermal_first.py
single_fasterrcnn_7_23.pth  ---> Trained by Student_Fasterrcnn_r18_fpn_1x_kaist_amfd.py
```


## Inference
We take the example of inference on the KAIST dataset:
1. Download checkpoints you need for test. You can refer to [Datasets and Models](#datasets-and-models). Here we need the checkpoint "resnet18-5c106cde.pth" and "single_fasterrcnn_7_23.pth" to initialize the weights.


2. Modify the corresponding paths in the configuration file, including the file save path, the path of the checkpoint used for initialization, and the dataset path.

3. Inference.
	```
	cd mmdetection
	python ./tools/test.py ./projects/AMFD/config/KAIST/Student_Fasterrcnn_r18_fpn_1x_kaist.py {your_path}/single_fasterrcnn_7_23.pth
	```

## Distill
1. Download the checkpoint of teacher network. You can refer to [Datasets and Models](#datasets-and-models). Here we need the checkpoint "resnet18-5c106cde.pth" (for the student) and "Teacher_Fasterrcnn_7_66.pth" (as the teacher checkpoint) to initialize the weights.
If you want to train a teacher network by yourself:
	```
	python ./tools/train.py ./projects/AMFD/config/KAIST/Teacher_Fasterrcnn_r50_fpn_1x_kaist_thermal_first.py
	```

2. Modify paths in the config "Student_Fasterrcnn_r18_fpn_1x_kaist_amfd.py"

3. Distill
	```
	python ./tools/train.py ./projects/AMFD/config/KAIST/Student_Fasterrcnn_r18_fpn_1x_kaist_amfd.py
	```

## citation
If you find our AMFD useful, please cite our paper:
```
  @article{amfd,
  	title={AMFD: Distillation via Adaptive Multimodal Fusion for Multispectral Pedestrian Detection},
  	author={Chen, Zizhao and Qian, Yeqiang and Yang, xiaoxiao and Wang, Chunxiang and Yang, Ming},
  	journal={arXiv preprint arXiv:2405.12944},
  	year={2024}}
```


## Contact

email: czz-000@sjtu.edu.cn, qianyeqiang@sjtu.edu.cn, xiaoxiaoyang7021@gmail.com
