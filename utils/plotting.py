# """Util functions"""
import torch
import numpy as np

import matplotlib.pyplot as plt
def renorm(x):
    mx=x.max()
    mn=x.min()
    return (x-mn)/(mx-mn)

def show_image_annot(images,masks,showPics = 5,*args,**kwargs):
    imgs, msks = images[:showPics],masks[:showPics]
    print("BATCHSIZE=",len(msks))
    for i in (imgs,msks):
        assert i.__class__.__name__ == 'ndarray', 'input data type should be ndarray'
    
    print(imgs.shape,'\n',msks.shape)
    plt.figure(figsize=(showPics,2))
    for i, (img,msk) in enumerate(zip(imgs,msks)):
        plt.subplot(2,len(imgs),i+1)
        plt.imshow(img, *args,**kwargs);plt.axis("off")
        plt.subplot(2,len(msks),i+len(imgs)+1)
        plt.imshow(msk, *args,**kwargs);plt.axis("off")
    plt.show()
    plt.close()