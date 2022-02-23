import nibabel as nib
import numpy as np
import imageio
import os
import cv2
from PIL import Image
def convertpng(pngfile, outdir):
    try:
        image_file = Image.open(pngfile)
        image_file = image_file.convert('L')
        image_file.save(os.path.join(outdir, os.path.basename(pngfile)))
    except Exception as e:
        print(e)
def read_niifile(niifile):  # 读取niifile文件
    img = nib.load(niifile)  # 下载niifile文件（其实是提取文件）
    img_fdata = img.get_fdata()  # 获取niifile数据
    return img_fdata

def RotateClockWise90(img):   #图片逆向旋转90°，同时水平旋转。
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 0)
    new_img = cv2.flip(new_img, 1)
    return new_img

def save_fig(file, savepicdir):  # 保存为图片
    fdata = read_niifile(file)  # 调用上面的函数，获得数据
    (x, y, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量）

    for k in range(z):
        silce = fdata[:, :, k]  # 三个位置表示三个不同角度的切片
        # print(slice.size)
        imageio.imwrite(os.path.join(savepicdir, '{}.png'.format(k)), silce)
        # 将切片信息保存为png格式
        img1 = cv2.imread(os.path.join(savepicdir, '{}.png'.format(k)))
        img1 =RotateClockWise90(img1)
        '''
        for ii in range(512):
            for jj in range(512 - ii - 1):
                value = img1[ii, jj, 0]
                img1.itemset((ii, jj, 0), img1[511 - ii, 511 - jj, 0])
                img1.itemset((ii, jj, 1), img1[511 - ii, 511 - jj, 0])
                img1.itemset((ii, jj, 2), img1[511 - ii, 511 - jj, 0])
                img1.itemset((511 - jj, 511 - ii, 0), value)
                img1.itemset((511 - jj, 511 - ii, 1), value)
                img1.itemset((511 - jj, 511 - ii, 2), value)
                # convertpng(img1, img_dir2)
        '''
        cv2.imwrite(os.path.join(savepicdir, '{}.png'.format(k)), img1)
        convertpng(os.path.join(savepicdir, '{}.png'.format(k)), savepicdir)

def save_fig1(file, savepicdir):  # 保存为图片
    fdata = read_niifile(file)  # 调用上面的函数，获得数据
    (x, y, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量）

    for k in range(z):
        silce = fdata[:, :, k]  # 三个位置表示三个不同角度的切片
        # print(slice.size)
        imageio.imwrite(os.path.join(savepicdir, '{}_gt.png'.format(k)), silce)
        # 将切片信息保存为png格式
        img1 = cv2.imread(os.path.join(savepicdir, '{}_gt.png'.format(k)))
        img1 = RotateClockWise90(img1)
        '''
        for ii in range(512):
            for jj in range(512 - ii - 1):
                value = img1[ii, jj, 0]
                img1.itemset((ii, jj, 0), img1[511 - ii, 511 - jj, 0])
                img1.itemset((ii, jj, 1), img1[511 - ii, 511 - jj, 0])
                img1.itemset((ii, jj, 2), img1[511 - ii, 511 - jj, 0])
                img1.itemset((511 - jj, 511 - ii, 0), value)
                img1.itemset((511 - jj, 511 - ii, 1), value)
                img1.itemset((511 - jj, 511 - ii, 2), value)
                # convertpng(img1, img_dir2)
        '''
        cv2.imwrite(os.path.join(savepicdir, '{}_gt.png'.format(k)), img1)
        convertpng(os.path.join(savepicdir, '{}_gt.png'.format(k)), savepicdir)

def save_fig2(file, savepicdir):  # 保存为图片
    fdata = read_niifile(file)  # 调用上面的函数，获得数据
    (x, y, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量）

    for k in range(z):
        silce = fdata[:, :, k]  # 三个位置表示三个不同角度的切片
        # print(slice.size)
        imageio.imwrite(os.path.join(savepicdir, '{}_pre.png'.format(k)), silce)
        # 将切片信息保存为png格式
        img1 = cv2.imread(os.path.join(savepicdir, '{}_pre.png'.format(k)))
        img1 = RotateClockWise90(img1)
        '''
        for ii in range(512):
            for jj in range(512-ii-1):
                value = img1[ii, jj, 0]
                img1.itemset((ii, jj, 0), img1[511 - ii, 511 - jj, 0])
                img1.itemset((ii, jj, 1), img1[511 - ii, 511 - jj, 0])
                img1.itemset((ii, jj, 2), img1[511 - ii, 511 - jj, 0])
                img1.itemset((511 - jj, 511 - ii, 0), value)
                img1.itemset((511 - jj, 511 - ii, 1), value)
                img1.itemset((511 - jj, 511 - ii, 2), value)
                # convertpng(img1, img_dir2)
        '''
        cv2.imwrite(os.path.join(savepicdir, '{}_pre.png'.format(k)), img1)
        convertpng(os.path.join(savepicdir, '{}_pre.png'.format(k)), savepicdir)

# root_path = r'I:\两阶段实验\第一阶段\实验结果196_16'
root_path = r'./'
path_2D = os.path.join(root_path,'train')
path_3D = os.path.join(root_path, 'train_data')
for i in range(0,1):
    os.mkdir(path_2D+'/case_'+str(i).zfill(2))
    path1=os.path.join(path_2D, 'case_'+str(i).zfill(2))
    os.mkdir(path1+'/raw_data')
    os.mkdir(path1+'/pseudo_lable')
    os.mkdir(path1+'/truth')
    #转raw_data
    dir = os.path.join(path_3D,'validation_case_'+str(i).zfill(2),'data_raw-data.nii.gz')
    # dir =r'I:\两阶段实验\第一阶段\实验结果196_16\train_data\validation_case_00\data_raw-data.nii.gz'  # nii的路径
    savepicdir = os.path.join(path_2D,'case_'+str(i).zfill(2),'raw_data')
    # savepicdir = r'I:\两阶段实验\第一阶段\实验结果196_16\train\case_00\raw_data'
    save_fig(dir, savepicdir)

    #转truth
    dir = os.path.join(path_3D, 'validation_case_' + str(i).zfill(2), 'truth.nii.gz')
    # dir =r'I:\两阶段实验\第一阶段\实验结果196_16\train_data\validation_case_00\data_raw-data.nii.gz'  # nii的路径
    savepicdir = os.path.join(path_2D, 'case_' + str(i).zfill(2), 'truth')
    # savepicdir = r'I:\两阶段实验\第一阶段\实验结果196_16\train\case_00\raw_data'
    save_fig1(dir, savepicdir)

    #转prediction
    dir = os.path.join(path_3D, 'validation_case_' + str(i).zfill(2), 'prediction.nii.gz')
    # dir =r'I:\两阶段实验\第一阶段\实验结果196_16\train_data\validation_case_00\data_raw-data.nii.gz'  # nii的路径
    savepicdir = os.path.join(path_2D, 'case_' + str(i).zfill(2), 'pseudo_lable')
    # savepicdir = r'I:\两阶段实验\第一阶段\实验结果196_16\train\case_00\raw_data'
    save_fig2(dir, savepicdir)


