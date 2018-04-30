f=open(r'E:/MILdata/train.txt','r')
line= f.readlines()
for i in range(len(line)):
    if 'E:/code/Mytry/data' in line[i]:
        line[i]=line[i].replace('E:/code/Mytry/data','E:/MILdata')
open(r'E:/MILdata/train.txt','w').writelines(line)