# UnetCosLoss
🔥Cosine loss used in output of Unet for few-shot semantic segmentation

## Dataset:
* 🔥For kaggle competition: https://www.kaggle.com/competitions/aia-chaos-class-obdet
```
# Dataset with auxilary labels (torsal segmentation, marked as "trunk"), and annotation files
gdown https://drive.google.com/u/0/uc?id=1xgzM-eFbFprpaLEKvw9eTBAwVpS-xWRh&export=download
```
* Test files are not included, please see ["Data Description
"](https://www.kaggle.com/competitions/aia-chaos-class-obdet-da/data) in the competition webpage

* Structure of the dataset
```
root
|- CT
|   |-annotations
|   |   - annotations_subj{X}.json (X=1,2,5,6,8,10,14,16,18,19,21,22,23,24,25,26,27,28,29,30)
|   |-  {X} (X=1,2,5,6,8,10,14,16,18,19,21,22,23,24,25,26,27,28,29,30)
|   |   |-DICOM_anon
|   |   |   |-*.dcm
|   |   |-NewGT
|   |       |-*.png
|- MRI
|   |- annotations
|   |   |- MRI_Label
|   |   |   |- annotations_subj{X}.json (X=1,2,3,5,8)
|   |   |   |- annotations.json
|   |   |- MRI_NonLabel
|   |   |   |- annotations_subj{X}.json (X=10,13,15,19,20,21,22,31,32,33)
|   |   |   |- annotations.json
|   |- MRI_Label
|   |   |- {X} (X=1,2,3,5,8)
|   |      |-T2SPIR
|   |         |-DICOM_anon
|   |         |   |-*.dcm
|   |         |-NewGT
|   |             |-*.png
|   |- MRI_NonLabel
|       |- {X} (X=10,13,15,19,20,21,22,31,32,33)
|          |-T2SPIR
|             |-DICOM_anon
|                 |-*.dcm
#|- testset (MRI testset for kaggle competition)
#    |-annotations
#    |   - annotations_subj{X}.json (X=34,36,37,38,39)
#    |-  {X} (X=34,36,37,38,39)
#          |-T2SPIR
#              |-DICOM_anon
#                  |-*.dcm
```
## Requirements:
Required repos are in requirements.txt, install with:
```
bash setup.sh
```

## Let's Train the Model:
* Training: Try_Training.ipynb- Training processes of source dataset and target dataset
* Submission(for kaggle contest): Try_Submission.ipynb- running length encoding for submission
* Other things are explained below

## Data loader
```
root/dataloaders:
  - ctmri_multiclass.py: Tran/Test pytorch Dataset class, takes an anootation in and make Dataset(s) with/without tran-test split
Try_Data_Loader.ipynb: Try-out the data loader, demonstrate how data would be loaded (format, amount,...)
```

## Model/Loss:
```
root/models:
  - backbone.py: Unet
  - head.py: Output layer for Baseline++ Method with cosine similarity metic computed by feature map pixel
  - models.py: Constructing whole model for Baseline++ Method
root/losses:
  - marginal.py: Implementation of Focal loss (FocalLoss) and Cosine face loss (AddMarginLoss)
Try_Model_Shape.ipynb: Try-out the model and loss function, demonstrate how data would be loaded (format, type, amount, shape...)
```

## Citations
### Githubs
* Unet
```
@misc{Yakubovskiy:2019,
  Author = {Pavel Yakubovskiy},
  Title = {Segmentation Models Pytorch},
  Year = {2020},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
```
* AddMarginLoss/Focal Loss
```
@misc{ronghuaiyang:2019,
  Author = {ronghuaiyang},
  Title = {arcface-pytorch},
  Year = {2019},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/ronghuaiyang/arcface-pytorch}}
}
```
* Pycocotools
```
@misc{cocodataset:2015,
  Author = {cocodataset},
  Title = {cocoapi},
  Year = {2015},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/cocodataset/cocoapi}}
}
```
* Pytorch
```
@incollection{NEURIPS2019_9015,
title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {8024--8035},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf}
}
```

### Papers
Methods
* Chen, W. Y., Liu, Y. C., Kira, Z., Wang, Y. C. F., & Huang, J. B. (2019). A closer look at few-shot classification. arXiv preprint arXiv:1904.04232.
* Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., ... & Liu, W. (2018). Cosface: Large margin cosine loss for deep face recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5265-5274).
* Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).
* Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.

Dataset
* A.E. Kavur, N.S. Gezer, M. Barış, S. Aslan, P.-H. Conze, et al. "CHAOS Challenge - combined (CT-MR) Healthy Abdominal Organ Segmentation",  Medical Image Analysis, Volume 69, 2021. https://doi.org/10.1016/j.media.2020.101950
* A.E. Kavur, M. A. Selver, O. Dicle, M. Barış,  N.S. Gezer. CHAOS - Combined (CT-MR) Healthy Abdominal Organ Segmentation Challenge Data (Version v1.03) [Data set]. Apr.  2019. Zenodo. http://doi.org/10.5281/zenodo.3362844
* A.E. Kavur, N.S. Gezer, M. Barış, Y.Şahin, S. Özkan, B. Baydar, et al.  "Comparison of semi-automatic and deep learning-based automatic methods for liver segmentation in living liver transplant donors", Diagnostic and  Interventional  Radiology,  vol. 26, pp. 11–21, Jan. 2020. https://doi.org/10.5152/dir.2019.19025
