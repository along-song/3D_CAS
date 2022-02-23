
from surface_distance import metrics
import surface_distance
import numpy as np
import nibabel as nib
import os

path=r'/home/sy/Desktop/3DUnet-server/brats/hybridlossffr'

sum_distance=0
num=0
file_list=os.listdir(path)
fp=open(os.path.join(path,'average_surfance_distance.txt'),'w')
fp.write('case_name        average_surfance_distance   \n')

for file_path in file_list:
        # pre_path=os.path.join(path,file_path,file_path+'_pred.nii.gz')
        # truth_path=os.path.join(path,file_path,file_path+'_gt.nii.gz')
        pre_path = os.path.join(path, file_path, 'prediction.nii.gz')
        truth_path = os.path.join(path, file_path, 'truth.nii.gz')
        pre_img = nib.load(pre_path)
        truth_img = nib.load(truth_path)
        width, height, queue = pre_img.dataobj.shape
        mask_gt = np.zeros((width, height, queue), np.bool)
        mask_pred = np.zeros((width, height, queue), np.bool)

        mask_gt[:, :, :] = truth_img.dataobj[:, :, :]
        mask_pred[:, :, :] = pre_img.dataobj[:, :, :]


        surfance_distances = surface_distance.compute_surface_distances(
                mask_gt, mask_pred, spacing_mm=(1, 1, 1))

        average_surfance_distance = surface_distance.compute_average_surface_distance(surfance_distances)
        print(file_path,'average_surfance_distance[1]+average_surfance_distance[0]:  ',(average_surfance_distance[1]+average_surfance_distance[0])/2.0)
        sum_distance+=(average_surfance_distance[1]+average_surfance_distance[0])/2.0
        num+=1
        fp.write(file_path+'      '+str(average_surfance_distance[1])+'\n')

all_average_distance=sum_distance/num

fp.write('all_average_surfance_distance is    '+str(all_average_distance))
print('all_average_surfance_distance is  ',all_average_distance)

fp.close()






