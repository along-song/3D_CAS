from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
import numpy as np
import tensorflow as tf
from keras import backend as K
K.set_image_data_format("channels_first")
from keras.layers.merge import concatenate

def botttleneck_layer(input_x,filters=12,padding='same',strides=(1,1,1)):

    x=Conv3D(filters,(3,3,3),padding=padding,strides=strides)(input_x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    return x

def transition_layer(x,n_filters):
    x=Conv3D(n_filters,(1,1,1),padding='same',strides=[1,1,1])(x)
    x=BatchNormalization(axis=1)(x)
    x=Activation('relu')(x)
    return x
def Concatenation(layers):
    return concatenate(layers, axis=1)

def dense_block(input_x,nb_layers=4,filters=12):
    layers_concat=list()
    layers_concat.append(input_x)
    x=botttleneck_layer(input_x,filters=filters,)
    layers_concat.append(x)

    for i in range(nb_layers-1):
        x=Concatenation(layers_concat)
        x =botttleneck_layer(x,filters=filters)
        layers_concat.append(x)
    x=Concatenation(layers_concat)

    return x