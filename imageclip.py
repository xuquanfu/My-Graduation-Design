#coding: utf-8
import os
import cv2
import numpy as np
import json
import re
import scipy.misc


# img_path_root = r'E:/dataseg/Tiszadob_hist_specify_3_ch_2_to_1/'
# img_path_root = r'E:/dataseg/Tiszadob_GT/'
# img_sub = 'E:/dataseg/tiszadob_hist_specify112_gt/'
# img_sub = 'E:/dataseg/tiszadob_hist_specify112/'

img_path_root = r'E:/dataseg/Szada_Scene1/'
img_sub = 'E:/dataseg/Szada_Scene1_seg/'
img_dir_firsts = os.listdir(img_path_root)
#print(img_dir_firsts)
img_width = 112
img_height = 112

for img_dir_first in img_dir_firsts:
    img_dir_first = img_dir_first.strip().strip('.bmp')
    print(img_dir_first)
    i = re.sub('\D','',img_dir_first)[0] #i指的是字符串中的第一个数字 即表示场景几
    print(i)
    #print(type(img_dir_first))
    #print('Scene'+str(i))

    # img_sub_path = img_sub + 'tiszadob_Scene' + str(i)
    # #img_sub_path = img_sub + 'Szada_Scene' + str(i)
    # if not os.path.exists(img_sub_path):
    #     os.mkdir(img_sub_path)
    img_np = scipy.misc.imread(os.path.join(img_path_root,img_dir_first+'.bmp'))
    print (img_np.shape)
    #img_np = img_np[0:630,0:945]
    #scipy.misc.imsave(img_sub  + img_dir_first + '.bmp', img_np)
    #print(type(img_np))
    img_shape = np.shape(img_np)

    print(img_shape)
    img_num = 0
    for h in range(0,img_shape[0],112):
        for w in range(0,img_shape[1],112):
            if h > 630 or w >940:
                continue
            sub_img = img_np[h:min(h + img_height, img_shape[0]), w:min(w + img_width, img_shape[1])]
            print(sub_img.shape)

            if sub_img.shape[0] != img_height or sub_img.shape[1] != img_width:

                sub_img = np.lib.pad(sub_img, ((0, img_height - int(sub_img.shape[0])), (0, img_width - int(sub_img.shape[1])), (0, 0)), 'constant',
                                     constant_values=0)
                # sub_img = np.lib.pad(sub_img, (
                # (0, img_height - int(sub_img.shape[0])), (0, img_width - int(sub_img.shape[1]))), 'constant',
                #                      constant_values=0)
            print(sub_img.shape)
            # scipy.misc.imsave(img_sub_path+'/'+img_dir_first+'_'+'clip'+str(img_num)+'.bmp',sub_img)
            scipy.misc.imsave(img_sub + '/' + img_dir_first + '_' + 'clip' + str(img_num) + '.bmp', sub_img)
            #f1 = open(img_sub_path+'/'+'index.txt','a')
            #s=img_dir_first+'_'+'clip'+str(img_num)
            #f1.write(s+':'+'['+str(h)+','+str(w)+']'+'\n')
            img_num = img_num+1
            print (img_num)
    #print(img_sub_path)
