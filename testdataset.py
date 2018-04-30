import scipy.misc
import numpy as np

class DataProvide():
    def __init__(self):
        self.path=[]
        self.image = []
        self.data_dir = []
        self.pairdata_dir = []
        self.label_dir = []
        self.batch_size = 1
        self.current_step = 0
        self.load_dataset()




    """
    Load set of images in a directory.
    This will automatically allocate a
    random 20% of the images as a test set

    data_dir: path to directory containing images
    """

    def load_dataset(self):
        f=open(r'E:/code/Mytry/data/test_Szada_Scene1.txt', 'r')
        self.image=f.readlines()
        num=len(self.image)
        # np.random.shuffle(self.image)
        for i in range(num):
            self.data_dir.append(self.image[i].split()[0])
            self.pairdata_dir.append(self.image[i].split()[1])
            self.label_dir.append(self.image[i].split()[2])
        f.close()

    def next_batch(self):
        steps = len(self.image) // self.batch_size
        if (self.current_step > steps - 1):
            self.current_step = 0
            # np.random.shuffle(self.image)

        data_batch = []
        label_batch = []
        pairdata_batch=[]
        data_path_batch = self.data_dir[self.current_step * self.batch_size: (self.current_step + 1) * self.batch_size]
        pairdata_path_batch = self.pairdata_dir[self.current_step * self.batch_size: (self.current_step + 1) * self.batch_size]
        label_path_batch = self.label_dir[self.current_step * self.batch_size: (self.current_step + 1) * self.batch_size]

        for i in range(len(data_path_batch)):
            img_path = data_path_batch[i]
            pairimg_path = pairdata_path_batch[i]
            label_path = label_path_batch[i]
            img = scipy.misc.imread(img_path)
            pairimg=scipy.misc.imread(pairimg_path)
            label = scipy.misc.imread(label_path)
            data_batch.append(img)
            pairdata_batch.append(pairimg)
            label_batch.append(label)

        self.current_step += 1
        #data_batch=np.reshape(data_batch,112*112*3)
        #pairdata_batch = np.reshape(pairdata_batch,  112 * 112 * 3)
        #label_batch = np.reshape(label_batch, 112 * 112 * 1 )
        #print(np.shape(data_batch))
        #print(np.shape(pairdata_batch))
        #print(np.shape(label_batch))
        return data_batch, pairdata_batch,label_batch
'''
        for i in range(len(data_path_batch)):
            img_path = data_path_batch[i]
            pairimg_path = pairdata_path_batch[i]
            label_path = label_path_batch[i]

        self.current_step += 1

        return img_path, pairimg_path,label_path

data =DataProvide()
a,b,c=data.next_batch()
print(a)
print(b)
print(c)
a1,b1,c1=data.next_batch()
print(a1)
print(b1)
print(c1)
'''