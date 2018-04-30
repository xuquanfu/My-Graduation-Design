
# coding: utf-8

# In[17]:

import numpy as np
import scipy.misc




path = 'E:/code/Mytry/data/evalution/'
gt_label = scipy.misc.imread(path+'Szada_Scene1_gt_testregion.png')
out = scipy.misc.imread(path+'Szada_Scene1.bmp')
f = open(path+'evalution.txt','w')


F_A = [];
M_A = [];
O_E = [];

Pr = [];
Re = [];
F_measure = [];

F_A_update = [];
M_A_update = [];
O_E_update = [];

Pr_update = [];
Re_update = [];
F_measure_update = [];


iter = 1
print(np.shape(gt_label))
print(np.shape(out))

for i in range(0, iter) :

    ind = out
    
    a = 0.0  #true-pos
    b = 0.0 #false-pos
    c = 0.0  #false-neg
    d = 0.0  #true-neg

    for h in range(0,ind.shape[0]) :
        for w in range(0,ind.shape[1]) :
            if ((gt_label[h,w] == 255 or gt_label[h,w] == 1) and ind[h,w] == 255)  :
                a+=1.0;
            elif (gt_label[h,w] == 0 and ind[h,w] == 255 ) :
                b+=1.0;
            elif ((gt_label[h,w] == 255 or gt_label[h,w] == 1) and ind[h,w] == 0)  :
                c+=1.0;
            else :
                d+=1.0;
    F_A.append(b/(a+b+c+d)*100);
    M_A.append(c/(a+b+c+d)*100);
    O_E.append((b+c)/(a+b+c+d)*100);

    if (a!=0.0 or b!=0.0):
	    pr = (0.0+a)/(a+b);
	    Pr.append(pr*100);
    else :
	#Pr.append(0.0);
	    Pr.append(1.0*100);
    if (a!=0.0 or c!=0.0):
	    re = (0.0+a)/(a+c);
	    Re.append(re*100);
    else :
	    Re.append(1.0*100);
    if (a!=0.0):
    	F_measure.append(2*pr*re/(pr+re)*100);#2*Pr*Re/(Pr+Re);
    else :
	    F_measure.append(0.0);

    # a = 0.0  # true-pos
    # b = 0.0  # false-pos
    # c = 0.0  # false-neg
    # d = 0.0  # true-neg
    # for h in range(0, updatedMap.shape[0]):
    #     for w in range(0, updatedMap.shape[1]):
    #         if ((gt_label[h, w] == 255 or gt_label[h, w] == 1) and updatedMap[h, w] == 255):
    #             a += 1.0;
    #         elif (gt_label[h, w] == 0 and updatedMap[h, w] == 255):
    #             b += 1.0;
    #         elif ((gt_label[h, w] == 255 or gt_label[h, w] == 1) and updatedMap[h, w] == 0):
    #             c += 1.0;
    #         else:
    #             d += 1.0;
    # F_A_update.append(b / (a + b + c + d) * 100);
    # M_A_update.append(c / (a + b + c + d) * 100);
    # O_E_update.append((b + c) / (a + b + c + d) * 100);
    #
    # if (a != 0.0 or b != 0.0):
    #     pr = (0.0 + a) / (a + b);
    #     Pr_update.append(pr * 100);
    # else:
    #     # Pr.append(0.0);
    #     Pr_update.append(1.0 * 100);
    # if (a != 0.0 or c != 0.0):
    #     re = (0.0 + a) / (a + c);
    #     Re_update.append(re * 100);
    # else:
    #     Re_update.append(1.0 * 100);
    # if (a != 0.0):
    #     F_measure_update.append(2 * pr * re / (pr + re) * 100);  # 2*Pr*Re/(Pr+Re);
    # else:
    #     F_measure_update.append(0.0);



f.write('before update:\n')
f.write('F_A:\n')
f.write(str(F_A))
f.write('\n')
print ('F_A:')
print (F_A)

f.write('M_A:\n')
f.write(str(M_A))
f.write('\n')
print ('M_A')
print (M_A)

f.write('O_E:\n')
f.write(str(O_E))
f.write('\n')
print ('O_E')
print (O_E)

f.write('Pr:\n')
f.write(str(Pr))
f.write('\n')
print ('Pr:')
print (Pr)

f.write('Re:\n')
f.write(str(Re))
f.write('\n')
print ('Re')
print (Re)

f.write('F_measure:\n')
f.write(str(F_measure))
f.write('\n')
print ('F-measure')
print (F_measure)

f.close()
# f1.write('after update:\n')
# f1.write('F_A_update:\n')
# f1.write(str(F_A_update))
# f1.write('\n')
# print ('F_A_update:')
# print (F_A_update)
#
# f1.write('M_A_update:\n')
# f1.write(str(M_A_update))
# f1.write('\n')
# print ('M_A_update')
# print (M_A_update)
#
# f1.write('O_E_update:\n')
# f1.write(str(O_E_update))
# f1.write('\n')
# print ('O_E_update')
# print (O_E_update)
#
# f1.write('Pr_update:\n')
# f1.write(str(Pr_update))
# f1.write('\n')
# print ('Pr_update:')
# print (Pr_update)
#
# f1.write('Re_update:\n')
# f1.write(str(Re_update))
# f1.write('\n')
# print ('Re_update')
# print (Re_update)
#
# f1.write('F_measure_update:\n')
# f1.write(str(F_measure_update))
# f1.write('\n')
# print ('F-measure_update')
# print (F_measure_update)
# f1.close()
# print ('Success!')


                    
            
    
    
    



