from functools import partial
import tensorflow as tf
from keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1., tr=0.5):
    y_true_f = K.flatten(y_true)
    # y_true_f[y_true_f < tr] = 0
    # y_true_f[y_true_f >= tr] = 1
    y_pred_f = K.flatten(y_pred)
    # y_pred_f[y_pred_f < tr] = 0
    # y_pred_f[y_pred_f >= tr] = 1
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # sens = intersection / (intersection + (K.sum(y_true_f*(1-y_pred_f))) + 1e-6)
    # total = dice+0.1*sens
    return dice


def hybrid_loss(y_true, y_pred):
    return (dice_coef_loss(y_true, y_pred)+Tversky_loss(y_true, y_pred)+weighted_dice_coefficient_loss(y_true, y_pred))/3.0

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return intersection/(intersection+(y_true_f*(1-y_pred_f))+1e-6)

def recall_loss(y_true, y_pred):
    return -recall(y_true, y_pred)

def Tversky_loss_coefficient(y_true, y_pred, smooth=1., alpha=0.7, beta=0.3):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    num_fp=K.sum((1-y_true_f)*y_pred_f)
    num_fn=K.sum((1-y_pred_f)*y_true_f)
    tversky_loss=(2.0 *intersection+smooth)/(2.0*intersection+alpha*num_fp+beta*num_fn)
    return tversky_loss

def  Tversky_loss(y_true,y_pred):
    return -Tversky_loss_coefficient(y_true,y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
