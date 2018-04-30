import os
import cv2
import numpy as np
import json
import re
import scipy.misc

img_width = 112
img_height = 112
#path = r'/workspace/zmychange/dataset/tiszadob_hist_specify/outmap3/0.44'
path = r'E:\code\Mytry\data\output'

img_dir_firsts = os.listdir(path)
outmaplist = []
beoutmaplist = []
for img_dir_first in img_dir_firsts:
    # if 'updataoutmap_' in img_dir_first:
    if 'Tiszadob' in img_dir_first:
        outmaplist.append(img_dir_first)
    # if 'beforeoutmap_' in img_dir_first:
    #     beoutmaplist.append(img_dir_first)

fullimage = np.zeros([6*112,9*112],np.uint8)
# befullimage = np.zeros([6*112,9*112],np.uint8)
print (fullimage.shape)
num = 0
for i in range(0,fullimage.shape[0],112):
    for j in range(0,fullimage.shape[1],112):
        # index = outmaplist.index('updataoutmap_' + str(num) + '.jpg')
        index = outmaplist.index('Tiszadob' + str(num) + '.bmp')
        #print index
        fullimage[i:i + img_height , j:j + img_width] = scipy.misc.imread(path+'/'+outmaplist[index])
        # beindex = beoutmaplist.index('beforeoutmap_' + str(num) + '.jpg')
        # # print index
        # befullimage[i:i + img_height, j:j + img_width] = scipy.misc.imread(path + '/' + beoutmaplist[beindex])
        num = num+1

# sub_path = '/workspace/zmychange/dataset/Szada_hist_specify1/outmapfull1/resnet_deeplab10_weightall_triplet2b_30000'
sub_path = 'E:\code\Mytry\data\output'
if not os.path.exists(sub_path):
    os.mkdir(sub_path)
# sub_path2 = '/workspace/zmychange/dataset/Szada_hist_specify1/outmapfull1/resnet_deeplab10_weightall_triplet2b_30000/0.485'
#
# if not os.path.exists(sub_path2):
#     os.mkdir(sub_path2)
# scipy.misc.imsave(sub_path2+'/'+'updataSza_1.bmp',fullimage)
# scipy.misc.imsave(sub_path2+'/'+'beforeSza_1.bmp',befullimage)

fullim1_y = fullimage[0:448,0:784]
fullim2_for_knn=fullimage[0:452,0:788]
# befullim1_y = befullimage[0:640,0:952]
print (fullim1_y.shape)
# print (befullim1_y.shape)
# scipy.misc.imsave(sub_path2+'/'+'updataSza_1_fulloutmap.bmp',fullim1_y)
# scipy.misc.imsave(sub_path2+'/'+'beforeSza_1_fulloutmap.bmp',befullim1_y)
scipy.misc.imsave(sub_path+'/'+'Szada_Scene1.bmp',fullim1_y)
scipy.misc.imsave(sub_path+'/'+'Szada_Scene1_for_knn.bmp',fullim2_for_knn)