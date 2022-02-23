import nibabel as nib
from myvi import myvi
import numpy as np
import scipy.ndimage as ndimg

# get img data and spacing
nii = nib.load(r'path to/home/sy/桌面/Pancreas-CT(result)/Unet/prediction/validation_case_6/prediction.nii.gz')
imgs = nii.get_data()
zoom = nii.header.get_zooms()
# smooth (may loss details)
organ_1 = ndimg.gaussian_filter(np.float32(imgs==1), 1)
organ_2 = ndimg.gaussian_filter(np.float32(imgs==2), 1)
organ_3 = ndimg.gaussian_filter(np.float32(imgs==6), 1)

# vts, fs, ns, cs是节点，表面，法向量，颜色
vts, fs, ns, vs = myvi.util.build_surf3d(organ_1, 1, 0.5, zoom)
vts2, fs2, ns2, vs2 = myvi.util.build_surf3d(organ_2, 1, 0.5, zoom)
vts3, fs3, ns3, vs3 = myvi.util.build_surf3d(organ_3, 1, 0.5, zoom)

manager = myvi.Manager()
manager.add_surf('spleen', vts, fs, ns, (1,0,0))
manager.add_surf('pancreas', vts2, fs2, ns2, (0,1,0))
manager.add_surf('liver', vts3, fs3, ns3, (0,0,1))
manager.show('Pancrease 3D')