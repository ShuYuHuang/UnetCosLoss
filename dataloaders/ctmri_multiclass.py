from os.path import join

import cv2
import numpy as np

import torch
import torch.utils.data as tud

from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from pydicom import dcmread
### Utils
def load_dcm(fname):
    ds=dcmread(fname).pixel_array
    # 調整格式以配合albumentation套件需求
    return ds.astype('uint8')
def load_msk(fname):
    msk = cv2.imread(fname)[...,0]
    # 調整格式以配合albumentation套件需求
    return msk.astype('uint8')
def occr_in_ds(ds):
    rate_pos=np.count_nonzero(ds[0][1])/np.prod(ds[0][1].shape)
    cat_counts=[len(ds.coco_obj.getAnnIds(catIds=i)) for i in ds.cat_ids]
    occr_rate=[1-rate_pos]+[*map(lambda x:x*rate_pos/sum(cat_counts),cat_counts)]
    return occr_rate
### Dataset
class CTMRI_MultiClassDataset(tud.Dataset):
    def __init__(self,anno_file,
                 root_dir,
                 transform=None,
                 test_transform=None,
                 test_split=None):
        
        self.root_dir=root_dir
        self.transform=transform
        self.test_transform=test_transform
        self.coco_obj=COCO(anno_file)
        self.training=True
        if test_split:
            self.element_train,self.element_val=train_test_split(self.coco_obj.imgs,test_size=test_split)
        else:
            self.element_train=self.coco_obj.imgs
        self.n_cats=len(self.coco_obj.cats)
        self.cat_ids=list(self.coco_obj.cats.keys())
    def __len__(self) -> int:
        if self.training:
            return len(self.element_train)
        else:
            return len(self.element_val)
    def __getitem__(self,id):
        if self.training:
            img_obj=self.element_train[id]
            transform=self.transform
        else:
            img_obj=self.element_val[id]
            transform=self.test_transform
        # Read dcm to image
        image = load_dcm(join(self.root_dir,img_obj['file_name']))
        # read mask
        mask = load_msk(join(self.root_dir,img_obj['mask_file']))
        # Albumentation
        if transform:
            transformed = transform(image=image, mask=mask)
            image,mask = transformed['image'],transformed['mask']
        # Image preparation    
        image = image[np.newaxis,...].repeat(3,axis=0)
        # Mask preparation
        mapping=np.vectorize(lambda x: ([0]+self.cat_ids).index(x))
        mask=mapping(mask)
        return image,mask


## Dataset info
# === CT Sets ===
# --All labels of CT sets were reviewed. Some ground truth data might be slightly different from the first published set. Please use the last version of the sets.
# --All distinguishable "vena cava inferior" areas were excluded from the liver in ground truth data.
# --All gallbladder "vena cava inferior" areas were excluded from the liver in ground truth data.
# --Labeles of the four abdomen organs in the ground data are represented by four different pixel values ranges. These ranges are:
# Torsal: 32 (28<<<35)
# Liver: 63 (55<<<70)

# === MR Sets ===
# --All labels of MR sets were reviewed. Some ground truth data might be slightly different from the first published set. Please use the last version of the sets.
# --The In-phase and Out-phase images have same UID in the T1DUAL sequences. Therefore they were stored under two folder.
# --The ground images in T1DUAL folder represents both In-phase and Out-phase images.
# --The anonymization method of the MR sequences was changed to prevent UID data in DICOM images.
# --All distinguishable "vena cava inferior" areas were excluded from the liver in ground truth data.
# --All gallbladder "vena cava inferior" areas were excluded from the liver in ground truth data.
# --The shape of the kidneys are determined elliptical as much as possible. Veins, small artifacts are included to the kidney if they are inside of the kidneys elliptical contour.
# --Labeles of the four abdomen organs in the ground data are represented by four different pixel values ranges. These ranges are:
# Torsal: 32 (28<<<35)
# Liver: 63 (55<<<70)
# Right kidney: 126 (110<<<135)
# Left kidney: 189 (175<<<200)
# Spleen: 252 (240<<<255)

