#! /usr/bin/env python3

__version__= '1.0'

import argparse
import sys
import os
import numpy as np
import warnings
import xml.etree.ElementTree as et
import pandas as pd
from tqdm import tqdm
import csv
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow.keras.losses
from tensorflow.keras.layers import *

from tensorflow.python.keras import backend as tensorflow_backend






with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
__doc__=\
"""
Tool to load model and predict for given image.
"""

projection_dim = 64
patch_size = 1
num_patches =28*28
class Patches(layers.Layer):
    def __init__(self, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        print(tf.shape(images)[1],'images')
        print(self.patch_size,'self.patch_size')
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        print(patches.shape,patch_dims,'patch_dims')
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config
    
    
class PatchEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config
    

class sbb_predict:
    def __init__(self,image,model, patches='false',save='false', ground_truth=None,weights_dir=None ):
        self.image=image
        self.patches=patches
        self.save=save
        self.model_dir=model
        self.ground_truth=ground_truth
        self.weights_dir=weights_dir

    def resize_image(self,img_in,input_height,input_width):
        return cv2.resize( img_in, ( input_width,input_height) ,interpolation=cv2.INTER_NEAREST)
    
    
    def color_images(self,seg):
        ann_u=range(self.n_classes)
        if len(np.shape(seg))==3:
            seg=seg[:,:,0]
            
        seg_img=np.zeros((np.shape(seg)[0],np.shape(seg)[1],3)).astype(np.uint8)
        colors=sns.color_palette("hls", self.n_classes)
        
        for c in ann_u:
            c=int(c)
            segl=(seg==c)
            seg_img[:,:,0][seg==c]=c
            seg_img[:,:,1][seg==c]=c
            seg_img[:,:,2][seg==c]=c
        return seg_img
    
    def otsu_copy_binary(self,img):
        img_r=np.zeros((img.shape[0],img.shape[1],3))
        img1=img[:,:,0]

        #print(img.min())
        #print(img[:,:,0].min())
        #blur = cv2.GaussianBlur(img,(5,5))
        #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        retval1, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        

        img_r[:,:,0]=threshold1
        img_r[:,:,1]=threshold1
        img_r[:,:,2]=threshold1
        #img_r=img_r/float(np.max(img_r))*255
        return img_r
    
    def otsu_copy(self,img):
        img_r=np.zeros((img.shape[0],img.shape[1],3))
        #img1=img[:,:,0]

        #print(img.min())
        #print(img[:,:,0].min())
        #blur = cv2.GaussianBlur(img,(5,5))
        #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, threshold1 = cv2.threshold(img[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, threshold2 = cv2.threshold(img[:,:,1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, threshold3 = cv2.threshold(img[:,:,2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        

        img_r[:,:,0]=threshold1
        img_r[:,:,1]=threshold2
        img_r[:,:,2]=threshold3
        ###img_r=img_r/float(np.max(img_r))*255
        return img_r
    
    def soft_dice_loss(self,y_true, y_pred, epsilon=1e-6): 

        axes = tuple(range(1, len(y_pred.shape)-1))
        
        numerator = 2. * K.sum(y_pred * y_true, axes)
    
        denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
        return 1.00 - K.mean(numerator / (denominator + epsilon)) # average over classes and batch
    
    def weighted_categorical_crossentropy(self,weights=None):

        def loss(y_true, y_pred):
            labels_floats = tf.cast(y_true, tf.float32)
            per_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_floats,logits=y_pred)
        
            if weights is not None:
                weight_mask = tf.maximum(tf.reduce_max(tf.constant(
                    np.array(weights, dtype=np.float32)[None, None, None])
                    * labels_floats, axis=-1), 1.0)
                per_pixel_loss = per_pixel_loss * weight_mask[:, :, :, None]
            return tf.reduce_mean(per_pixel_loss)
        return self.loss


    def IoU(self,Yi,y_predi):
        ## mean Intersection over Union
        ## Mean IoU = TP/(FN + TP + FP)
    
        IoUs = []
        Nclass = np.unique(Yi)
        for c in Nclass:
            TP = np.sum( (Yi == c)&(y_predi==c) )
            FP = np.sum( (Yi != c)&(y_predi==c) )
            FN = np.sum( (Yi == c)&(y_predi != c)) 
            IoU = TP/float(TP + FP + FN)
            if self.n_classes>2:
                print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
            IoUs.append(IoU)
        if self.n_classes>2:
            mIoU = np.mean(IoUs)
            print("_________________")
            print("Mean IoU: {:4.3f}".format(mIoU))
            return mIoU
        elif self.n_classes==2:
            mIoU = IoUs[1]
            print("_________________")
            print("IoU: {:4.3f}".format(mIoU))
            return mIoU
            
    def start_new_session_and_model(self):
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.compat.v1.Session(config=config)  # tf.InteractiveSession()
        tensorflow_backend.set_session(session)
        #tensorflow.keras.layers.custom_layer = PatchEncoder
        #tensorflow.keras.layers.custom_layer = Patches
        self.model = load_model(self.model_dir , compile=False,custom_objects = {"PatchEncoder": PatchEncoder, "Patches": Patches})
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth=True
    
        #self.session = tf.InteractiveSession()
        #keras.losses.custom_loss = self.weighted_categorical_crossentropy
        #self.model = load_model(self.model_dir , compile=False)
        
        if self.weights_dir!=None:
            print('man burdayammmmaaa')
            self.model.load_weights(self.weights_dir)
            
        
        self.img_height=self.model.layers[len(self.model.layers)-1].output_shape[1]
        self.img_width=self.model.layers[len(self.model.layers)-1].output_shape[2]
        self.n_classes=self.model.layers[len(self.model.layers)-1].output_shape[3]

    def predict(self):
        self.start_new_session_and_model()
        if self.patches=='true' or self.patches=='True':
            print(self.patches,'gadaaiikk')
            #def textline_contours(img,input_width,input_height,n_classes,model):
            
            img=cv2.imread(self.image)
            
            img = self.otsu_copy_binary(img)
            
            #plt.imshow(img)
            #plt.show()
            
            if img.shape[0]<self.img_height:
                img=cv2.resize( img, ( img.shape[1],self.img_width) ,interpolation=cv2.INTER_NEAREST)

            if img.shape[1]<self.img_width:
                img=cv2.resize( img, ( self.img_height,img.shape[0]) ,interpolation=cv2.INTER_NEAREST)

            margin=True
            
            if margin:
                kernel = np.ones((5,5),np.uint8)
                img_width_model=self.img_width
                img_height_model=self.img_height
                
                
                margin = int(0.2 * img_width_model)

                width_mid = img_width_model - 2 * margin
                height_mid = img_height_model - 2 * margin


                img = img / float(255.0)

                img_h = img.shape[0]
                img_w = img.shape[1]

                prediction_true = np.zeros((img_h, img_w, 3))
                mask_true = np.zeros((img_h, img_w))
                nxf = img_w / float(width_mid)
                nyf = img_h / float(height_mid)

                if nxf > int(nxf):
                    nxf = int(nxf) + 1
                else:
                    nxf = int(nxf)

                if nyf > int(nyf):
                    nyf = int(nyf) + 1
                else:
                    nyf = int(nyf)

                for i in range(nxf):
                    for j in range(nyf):

                        if i == 0:
                            index_x_d = i * width_mid
                            index_x_u = index_x_d + img_width_model
                        elif i > 0:
                            index_x_d = i * width_mid
                            index_x_u = index_x_d + img_width_model

                        if j == 0:
                            index_y_d = j * height_mid
                            index_y_u = index_y_d + img_height_model
                        elif j > 0:
                            index_y_d = j * height_mid
                            index_y_u = index_y_d + img_height_model

                        if index_x_u > img_w:
                            index_x_u = img_w
                            index_x_d = img_w - img_width_model
                        if index_y_u > img_h:
                            index_y_u = img_h
                            index_y_d = img_h - img_height_model
                            
                        

                        img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]

                        label_p_pred = self.model.predict(
                            img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                        seg = np.argmax(label_p_pred, axis=3)[0]
                        
                        #seg = label_p_pred[0,:,:,3]
                        
                        #seg[seg>0.1] =1
                        #seg[seg<1] =0

                        seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                        if i==0 and j==0:
                            seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                            seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                            mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                            prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                            :] = seg_color
                            
                        elif i==nxf-1 and j==nyf-1:
                            seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                            seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]

                            mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                            prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0,
                            :] = seg_color
                            
                        elif i==0 and j==nyf-1:
                            seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                            seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]

                            mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                            prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin,
                            :] = seg_color
                            
                        elif i==nxf-1 and j==0:
                            seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                            seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                            mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                            prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                            :] = seg_color
                            
                        elif i==0 and j!=0 and j!=nyf-1:
                            seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                            seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                            mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                            prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                            :] = seg_color
                            
                        elif i==nxf-1 and j!=0 and j!=nyf-1:
                            seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                            seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                            mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                            prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                            :] = seg_color
                            
                        elif i!=0 and i!=nxf-1 and j==0:
                            seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                            seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                            mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                            prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                            :] = seg_color
                            
                        elif i!=0 and i!=nxf-1 and j==nyf-1:
                            seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                            seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]

                            mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                            prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin,
                            :] = seg_color

                        else:
                            seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                            seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                            mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                            prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                            :] = seg_color

                prediction_true = prediction_true.astype(np.uint8)
            
            
            
            
                

                ###width=self.img_width
                ###height=self.img_height

                ####offset=int(.1*width)
                ###offset=int(0.2*width)

                ###width_mid=width-2*offset
                ###height_mid=height-2*offset
                ####img= cv2.medianBlur(img, 5)
                ####img=self.otsu_copy_binary(img)
                ####img = cv2.GaussianBlur(img,(5,5),0)
                ####img=img.astype(np.uint8)
                ####for i in range(3):
                    ####img= cv2.medianBlur(img, 3)

                ###img=img/255.0


                ###img_h=img.shape[0]
                ###img_w=img.shape[1]

                ###prediction_true=np.zeros((img_h,img_w,3))
                ###mask_true=np.zeros((img_h,img_w))
                ###nxf=img_w/float(width_mid)
                ###nyf=img_h/float(height_mid)

                ###if nxf>int(nxf):
                    ###nxf=int(nxf)+1
                ###else:
                    ###nxf=int(nxf)
                    
                ###if nyf>int(nyf):
                    ###nyf=int(nyf)+1
                ###else:
                    ###nyf=int(nyf)

                ###for i in range(nxf):
                    ###for j in range(nyf):

                        ###if i==0:
                            ###index_x_d=i*width_mid
                            ###index_x_u=index_x_d+width#(i+1)*width
                        ###elif i>0:
                            ###index_x_d=i*width_mid
                            ###index_x_u=index_x_d+width#(i+1)*width

                        ###if j==0:
                            ###index_y_d=j*height_mid
                            ###index_y_u=index_y_d+height#(j+1)*height
                        ###elif j>0:
                            ###index_y_d=j*height_mid
                            ###index_y_u=index_y_d+height#(j+1)*height

                        ###if index_x_u>img_w:
                            ###index_x_u=img_w
                            ###index_x_d=img_w-width
                        ###if index_y_u>img_h:
                            ###index_y_u=img_h
                            ###index_y_d=img_h-height


                        ###img_patch=img[index_y_d:index_y_u,index_x_d:index_x_u,:]



                        ###label_p_pred=self.model.predict(
                            ###img_patch.reshape(1,img_patch.shape[0],img_patch.shape[1],img_patch.shape[2]))


                        ###seg=np.argmax(label_p_pred,axis=3)[0]

                        ###seg_color=self.color_images(seg)



                        ###seg_color=seg_color[offset:seg_color.shape[0]-offset,offset:seg_color.shape[1]-offset,:]
                        ###seg=seg[offset:seg.shape[0]-offset,offset:seg.shape[1]-offset]


                        ###mask_true[index_y_d+offset:index_y_u-offset,index_x_d+offset:index_x_u-offset]=seg
                        ###prediction_true[index_y_d+offset:index_y_u-offset,index_x_d+offset:index_x_u-offset,:]=seg_color
                        
                y_predi = mask_true

                #print(np.unique(mask_true))
                #find_contours(mask_true)

                #y_testi = label[:,:,0]#np.argmax(label.reshape(1,label.shape[0],label.shape[1],label.shape[2]), axis=3)



                #y_predi=cv2.erode(y_predi,kernel,iterations=3)
                y_predi=cv2.resize( y_predi, ( img.shape[1],img.shape[0]) ,interpolation=cv2.INTER_NEAREST)
                return y_predi
                        

            if not margin:

                kernel = np.ones((5,5),np.uint8)

                width=self.img_width
                height=self.img_height

                
                ###img = cv2.medianBlur(img,5)
                img=self.otsu_copy_binary(img)
                
                #img=cv2.bilateralFilter(img,9,75,75)
                ##img = cv2.GaussianBlur(img,(5,5),0)
                
                

                img=img/255.0
                


                img_h=img.shape[0]
                img_w=img.shape[1]

                prediction_true=np.zeros((img_h,img_w,3))
                mask_true=np.zeros((img_h,img_w))
                nxf=img_w/float(width)
                nyf=img_h/float(height)

                if nxf>int(nxf):
                    nxf=int(nxf)+1
                else:
                    nxf=int(nxf)
                    
                if nyf>int(nyf):
                    nyf=int(nyf)+1
                else:
                    nyf=int(nyf)

                print(nxf,nyf)
                for i in range(nxf):
                    for j in range(nyf):
                        index_x_d=i*width
                        index_x_u=(i+1)*width

                        index_y_d=j*height
                        index_y_u=(j+1)*height

                        if index_x_u>img_w:
                            index_x_u=img_w
                            index_x_d=img_w-width
                        if index_y_u>img_h:
                            index_y_u=img_h
                            index_y_d=img_h-height


                        img_patch=img[index_y_d:index_y_u,index_x_d:index_x_u,:]




                        label_p_pred=self.model.predict(img_patch.reshape(1,img_patch.shape[0],img_patch.shape[1],img_patch.shape[2]) )



                        seg=np.argmax(label_p_pred,axis=3)[0]
                        

                        seg_color=self.color_images(seg)

                        ###seg_color=color_images_diva(seg,n_classes)


                        mask_true[index_y_d:index_y_u,index_x_d:index_x_u]=seg
                        prediction_true[index_y_d:index_y_u,index_x_d:index_x_u,:]=seg_color




                    #mask_true=color_images(mask_true,n_classes)
                y_predi = mask_true

                #print(np.unique(mask_true))
                #find_contours(mask_true)

                #y_testi = label[:,:,0]#np.argmax(label.reshape(1,label.shape[0],label.shape[1],label.shape[2]), axis=3)



                #y_predi=cv2.erode(y_predi,kernel,iterations=3)
                y_predi=cv2.resize( y_predi, ( img.shape[1],img.shape[0]) ,interpolation=cv2.INTER_NEAREST)
                return y_predi

        #def extract_page(img,input_width,input_height,n_classes,model):
        if self.patches=='false' or self.patches=='False':

            img=cv2.imread(self.image)
            img_org_height=img.shape[0]
            img_org_width=img.shape[1]
            kernel = np.ones((5,5),np.uint8)

            width=self.img_width
            height=self.img_height
            
            #img = cv2.medianBlur(img,5)
            img=img.astype(np.uint8)
            #img=self.otsu_copy_binary(img)
            #img=img.astype(np.uint8)
            #img = cv2.medianBlur(img,5)
            img=img/255.0
            img=self.resize_image(img,self.img_height,self.img_width)
            

            label_p_pred=self.model.predict(
                img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))

            seg=np.argmax(label_p_pred,axis=3)[0]
            print(np.shape(seg),np.unique(seg))
            
            plt.imshow(seg*255)
            plt.show()
            seg_color=self.color_images(seg)
            print(np.shape(seg_color),np.unique(seg_color),'seg_color')


            #imgs = seg_color#/np.max(seg_color)*255#np.repeat(seg_color[:, :, np.newaxis], 3, axis=2)
            


            y_predi=cv2.resize( seg_color, ( img_org_width,img_org_height) ,interpolation=cv2.INTER_NEAREST)
            return y_predi[:,:,0]



    def run(self):
        res=self.predict()
        if self.save!=None:
            img=np.repeat(res[:, :, np.newaxis]*255, 3, axis=2)
            cv2.imwrite(self.save,img)
            
        if self.ground_truth!=None:
            gt_img=cv2.imread(self.ground_truth)
            print(np.shape(gt_img),np.shape(res))
            self.IoU(gt_img[:,:,0],res)
        print(np.unique(res))
        plt.imshow(res)
        plt.show()
def main():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-i','--image', dest='inp1', default=None, help='directory of alto files which have to be transformed.')
    parser.add_argument('-p','--patches', dest='inp3', default=False, help='use patches of image for prediction or should image resize be applied to be fit for model. this parameter should be true or false')
    parser.add_argument('-s','--save', dest='inp4', default=None, help='save prediction as a png file in current folder. this parameter should be false or true.')
    parser.add_argument('-m','--model', dest='inp2', default=None, help='model directory and name should be provided here.')
    parser.add_argument('-gt','--groundtruth', dest='inp5', default=None, help='ground truth directory if you want to see the iou of prediction.')
    parser.add_argument('-mw','--model_weights', dest='inp6', default=None, help='previous model weights which are saved.')
    
    options=parser.parse_args()
    
    possibles=globals()
    possibles.update(locals())
    x=sbb_predict(options.inp1,options.inp2,options.inp3,options.inp4,options.inp5,options.inp6)
    x.run()

if __name__=="__main__":
    main()

    
    
    
