from fastai.imports import *
from fastai import *
from fastai.vision import *
from torchvision.models import *
from sklearn.metrics import roc_curve, auc
import torch
import random
import PIL
import os
import numpy as np
from fastai.callbacks import *
from albumentations import (
    ElasticTransform,
    GridDistortion, MedianBlur, ToGray, JpegCompression, Transpose, ShiftScaleRotate, Normalize, 
    OpticalDistortion, Blur, CLAHE, RGBShift, ChannelShuffle, RandomContrast, RandomBrightness 
)
import warnings
warnings.filterwarnings('ignore')

from efficientnet_pytorch import EfficientNet

def tensor2np(x):
    np_image = x.cpu().permute(1, 2, 0).numpy()
    np_image = (np_image * 255).astype(np.uint8)
    
    return np_image

def alb_tfm2fastai(alb_tfm):
    def _alb_transformer(x):
        # tensor to numpy
        np_image = tensor2np(x)

        # apply albumentations
        transformed = alb_tfm(image=np_image)['image']

        # back to tensor
        tensor_image = pil2tensor(transformed, np.float32)
        tensor_image.div_(255)

        return tensor_image

    transformer = TfmPixel(_alb_transformer)
    
    return transformer()
    
class Auto_Image_Processing:
  def __init__(self):
    pass
  def plot_table(self, table):
  	fig, axs = plt.subplots(1,1, figsize=(15,2))
  	collabel=tuple(table.columns)
  	axs.axis('tight')
  	axs.axis('off')
  	the_table = axs.table(cellText=table.values,colLabels=collabel,loc='upper center')
  	return fig

  def seed_load(self,seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

  def get_model(self,model_name,pretrained=True, **kwargs):
    model = EfficientNet.from_pretrained(model_name)
    #model._fc = nn.Linear(model._fc.in_features, data.c)
    return model
  def res_dense_vgg(self,data,names,epochs):      
    fbeta = FBeta(average='weighted', beta = 1)
    model = cnn_learner(data, names,metrics=[AUROC(), FBeta(), accuracy])
    model.fit_one_cycle(epochs, max_lr =[8e-6, 8e-4, 8e-3] )#slice(8e-6, 8e-3)

    acc,roc=self.metrics_(model)
    return acc,roc,str(names),model
  def metrics_(self,learn):
    preds,y, loss = learn.get_preds(with_loss=True)
    acc = accuracy(preds, y)
    print(f'The accuracy is {acc*100} %.')
    probs = np.exp(preds[:,1])
    fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(f'ROC area is {roc_auc}\n')
    return acc.numpy(),roc_auc
  def eff(self,data,names_eff,epochs):
    model = Learner(data, self.get_model(names_eff), 
                metrics=[AUROC(), FBeta(), accuracy],
                wd=0.1,
                path = '.')
    model.fit(epochs,lr=1e-3)

    acc,roc=self.metrics_(model)
    return acc,roc,str(names_eff),model

  def Model_Names(self):
    """
    Call this function to get the list of available models
    """
    models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152','densenet121', 
              'densenet169','densenet201','densenet161','vgg16_bn', 'vgg19_bn','efficientnet-b0',
              'efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4',
              'efficientnet-b5','efficientnet-b6','efficientnet-b7']
    print('List of available models')
    for model in models:
      print(model)

  def imgClassificationcsv(self,img_path,train_path,model_list = None,tfms=True,advance_augmentation=False,
                           size=224,bs=16,epochs=1,test_size=0.3):
    """
        RETURNS MODEL
    
        model_list: list of names of all models that need to be trained 
          Call Model_Names() function to get the list of available models
        img_path: path to the images root folder containing the set of images
        train_path:path to the .csv file containing image names and labels
        tfms: augmentation transforms. Possible methods are True or False
        advance_augmentation: Should we apply advance data augmentation?
        size: individual image size in dataloader. Default size set to 224
        bs:batch size
        epochs:number of epochs for which the individual models need to be trained
        test_size:test size for ranking the models
        
    """
    dictionary={}
    dictionary["models"]=[resnet18, resnet34, resnet50, resnet101, resnet152,densenet121, densenet169,densenet201,densenet161,vgg16_bn, vgg19_bn]
    dictionary["model_names"]=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152','densenet121', 'densenet169','densenet201','densenet161','vgg16_bn', 'vgg19_bn']
    dictionary["efficientnets"]=["efficientnet-b{}".format(i) for i in range(0,8)]
    if model_list is None:
        model_list = dictionary["model_names"]
    path=''
    accuracy=[]
    roc_auc=[]
    list_names=[]
    train=pd.read_csv(train_path)
    
    if tfms==True:
        if advance_augmentation:
            tfms = get_transforms(do_flip = True, max_lighting = 0.2, max_zoom= 1.1, max_warp = 0.15, max_rotate = 45, 
                              xtra_tfms = [alb_tfm2fastai(ElasticTransform()), 
                                      alb_tfm2fastai(GridDistortion()), 
                                    alb_tfm2fastai(OpticalDistortion()),
                                      alb_tfm2fastai(MedianBlur()), 
                                      alb_tfm2fastai(ToGray()), 
                                    alb_tfm2fastai(JpegCompression()),
                                      alb_tfm2fastai(Transpose()), 
                                      alb_tfm2fastai(ShiftScaleRotate()), 
                                    alb_tfm2fastai(Normalize()),
                                      alb_tfm2fastai(Blur()), 
                                      alb_tfm2fastai(CLAHE()), 
                                    alb_tfm2fastai(RGBShift()),
                                      alb_tfm2fastai(ChannelShuffle()), 
                                      alb_tfm2fastai(RandomContrast()), 
                                      alb_tfm2fastai(RandomBrightness())])
        else:
            tfms = get_transforms(do_flip = True, max_lighting = 0.2, max_zoom= 1.1, max_warp = 0.15, max_rotate = 45)
    else:
        tfms=False
    data = ImageDataBunch.from_csv(path, folder= img_path, 
                              valid_pct = test_size,
                              csv_labels = train_path,
                              ds_tfms = tfms, 
                              fn_col = train.columns[0],
                              #test = 'train_SOaYf6m/images', 
                              label_col = train.columns[1],
                              bs = bs,
                              size = size,num_workers=0).normalize(imagenet_stats)
    #data.show_batch(rows=3, figsize=(5,5))
    print("Loaded dataset\n")
    print("Labels classified from the source are {}\n".format(data.classes))
    max_score=0
    for model in model_list:
        if model in dictionary["model_names"]:
            print("Started training {}\n".format(model))
            acc,auroc,names,model=self.res_dense_vgg(data,eval(model),epochs)
            accuracy.append(acc)
            roc_auc.append(auroc)
            list_names.append(names)
        if model in dictionary['efficientnets']:
            print("Started training {}\n".format(model))
            acc,auroc,names,model=self.eff(data,model,epochs)
            accuracy.append(acc)
            roc_auc.append(auroc)
            list_names.append(names)
        if acc>max_score:
            best_model=model
            max_score=acc
      
        else:
            del model
    df = pd.DataFrame(list(zip(model_list, accuracy,roc_auc)), 
              columns =['Model', 'Accuracy','ROAUC'])
    df.sort_values('Accuracy',inplace=True,ascending=False)
    self.plot_table(df)
    return best_model



  def imgClassificationfolder(self,img_path,model_list = None,tfms=True,advance_augmentation=False,
                              size=224,bs=16,epochs=1,test_size=0.3):
        """
        RETURNS MODEL
    
        model_list: list of names of all models that need to be trained 
          Call Model_Names() function to get the list of available models
        img_path: path to the images root folder containing the individual image folders
        tfms: augmentation transforms. Possible methods are True or False
        advance_augmentation: Should we apply advance data augmentation?
        size: individual image size in dataloader. Default size set to 224
        bs:batch size
        epochs:number of epochs for which the individual models need to be trained
        test_size:test size for ranking the models
        """
        dictionary={}
        dictionary["models"] = [resnet18, resnet34, resnet50, resnet101, resnet152,densenet121, 
                             densenet169,densenet201,densenet161,vgg16_bn, vgg19_bn]
        dictionary["model_names"] = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                  'densenet121', 'densenet169','densenet201','densenet161','vgg16_bn', 'vgg19_bn']
        dictionary["efficientnets"] = ["efficientnet-b{}".format(i) for i in range(0,8)]
        accuracy=[]
        roc_auc=[]
        list_names=[]
        path=''
        if model_list is None: 
            model_list = dictionary["model_names"]
        if tfms==True:
            if advance_augmentation:
                tfms = get_transforms(do_flip = True, max_lighting = 0.2, max_zoom= 1.1, max_warp = 0.15, max_rotate = 45, 
                                      xtra_tfms = [alb_tfm2fastai(ElasticTransform()), 
                                              alb_tfm2fastai(GridDistortion()), 
                                            alb_tfm2fastai(OpticalDistortion()),
                                              alb_tfm2fastai(MedianBlur()), 
                                              alb_tfm2fastai(ToGray()), 
                                            alb_tfm2fastai(JpegCompression()),
                                              alb_tfm2fastai(Transpose()), 
                                              alb_tfm2fastai(ShiftScaleRotate()), 
                                            alb_tfm2fastai(Normalize()),
                                              alb_tfm2fastai(Blur()), 
                                              alb_tfm2fastai(CLAHE()), 
                                            alb_tfm2fastai(RGBShift()),
                                              alb_tfm2fastai(ChannelShuffle()), 
                                              alb_tfm2fastai(RandomContrast()), 
                                              alb_tfm2fastai(RandomBrightness())])
            else:
                tfms = get_transforms(do_flip = True, max_lighting = 0.2, max_zoom= 1.1, max_warp = 0.15, max_rotate = 45)
        else:
            tfms=False
        data = ImageDataBunch.from_folder(img_path, valid_pct=test_size,ds_tfms=tfms, size=224, num_workers=4).normalize(imagenet_stats)
        print("Loaded dataset\n")
        print("Labels classified from the source are {}\n".format(data.classes))
        max_score=0
        for model in model_list:
            if model in dictionary["model_names"]:
                print("Started training {}\n".format(model))
                acc,auroc,names,model=self.res_dense_vgg(data,eval(model),epochs)
                accuracy.append(acc)
                roc_auc.append(auroc)
                list_names.append(names)
            if model in dictionary['efficientnets']:
                print("Started training {}\n".format(model))
                acc,auroc,names,model=self.eff(data,model,epochs)
                accuracy.append(acc)
                roc_auc.append(auroc)
                list_names.append(names)
            if acc>max_score:
                best_model=model
                max_score=acc
            else:
                del model
        df = pd.DataFrame(list(zip(model_list, accuracy,roc_auc)), 
                  columns =['Model', 'Accuracy','ROAUC'])
        df.sort_values('Accuracy',inplace=True,ascending=False)
        self.plot_table(df)
        return best_model