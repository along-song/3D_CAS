

from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D,add
from keras import backend as K
from keras import regularizers

def conv3d_bn(input_x,filters,kernel_size=(3,3,3),strides=(1,1,1),padding='same'):

    x=Conv3D(filters,kernel_size,padding=padding,strides=strides)(input_x)
    x=BatchNormalization(axis=1)(x)
    x=Activation('relu')(x)

    return x

def shortcut(input,residual):

    input_shape=K.int_shape(input)
    residual_shape=K.int_shape(residual)
    stride_length=int(round(input_shape[1]/residual_shape[1]))
    stride_width=int(round(input_shape[2]/residual_shape[2]))
    stride_height=int(round(input_shape[3]/residual_shape[3]))
    equal_channels=input_shape[0]==residual_shape[0]

    identity=input
    if stride_height!=1 or stride_length!=1 or stride_width!=1 or not equal_channels:
        identity=Conv3D(filters=residual_shape[0],kernel_size=(1,1,1),strides=(stride_width,stride_length,stride_height),padding="valid",
                        kernel_regularizer=regularizers.l2(0.0001))(input)

    return add([identity,residual])

def basic_block(filters,strides=(1,1,1)):

    def f(input):
        conv1 = conv3d_bn(input, filters, kernel_size=(3, 3, 3), strides=strides)
        conv2 = conv3d_bn(conv1, filters, kernel_size=(3, 3, 3), strides=strides)
        residual = conv3d_bn(conv2, filters, kernel_size=(3, 3, 3))

        return shortcut(input,residual)
    return f

def residual_block(input_x,filters=48,resid_num=1):

    res_input=basic_block(filters)(input_x)
    for i in range(resid_num-1):
        res_input=basic_block(filters)(res_input)

    return res_input

