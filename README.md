# UnetCosLoss
Cosine loss used in output of Unet for few shot semantic segmentation

## Dataset:
for kaggle competition: https://www.kaggle.com/competitions/aia-chaos-class-obdet
```
# Dataset with auxilary labels (torsal segmentation, marked as "trunk"), and annotation files
gdown https://drive.google.com/u/0/uc?id=1xgzM-eFbFprpaLEKvw9eTBAwVpS-xWRh&export=download

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

## Let's Train the Model:
```
Try_Training.ipynb: Training processes of source dataset and target dataset
```

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


