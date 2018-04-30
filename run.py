
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D) 
into a point in 2D.

By Youngwook Paul Kwon (young at berkeley.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt;
import scipy.misc


#import helpers
import inference
import visualize
import dataset
import testdataset
# prepare data and tf.session
#mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
data=dataset.DataProvide()
testdata=testdataset.DataProvide()
sess = tf.InteractiveSession()

# setup siamese network
siamese = inference.siamese();
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.005
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,1000, 0.90, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(siamese.loss,global_step=global_step)
saver = tf.train.Saver()
tf.global_variables_initializer().run()

# if you just want to load a previously trainmodel?
new = True
# model_ckpt = './model.ckpt'
# if os.path.isfile(model_ckpt):
input_var = None
while input_var not in ['yes', 'no']:
    input_var = input("We found model.ckpt file. Do you want to load it [yes/no]?")
if input_var == 'yes':
    new = False

# start training
if new:
    for step in range(500001):
        '''
        batch_x1, batch_y1 = mnist.train.next_batch(128)
        batch_x2, batch_y2 = mnist.train.next_batch(128)
        batch_y = (batch_y1 == batch_y2).astype('float')

        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x1: batch_x1,
                            siamese.x2: batch_x2,
                            siamese.y_: batch_y})
        '''
        batch_x1,batch_x2,batch_y=data.next_batch()
        _, loss_v,output_v,y_v = sess.run([train_step, siamese.loss,siamese.output,siamese.y_], feed_dict={
                            siamese.x1: batch_x1,
                            siamese.x2: batch_x2,
                            siamese.y_: batch_y})

        # print(y_v)
        # cv2.imshow('output',output_v)
        # cv2.waitKey(0)
        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if step % 50 == 0:
            print(output_v)
            print ('step %d: loss %.3f' % (step, loss_v))


        if step % 1000 == 0 and step > 0:
            saver.save(sess, './model.ckpt')

else:
    saver.restore(sess, './model100000.ckpt')
    for step in range(54):

        batch_x1,batch_x2,batch_y=testdata.next_batch()
        loss_v,output_v= sess.run([siamese.loss,siamese.output], feed_dict={
                            siamese.x1: batch_x1,
                            siamese.x2: batch_x2,
                            siamese.y_: batch_y})



        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()


        print ('step %d: loss %.3f' % (step, loss_v))
        print(output_v)
        output = np.zeros([112, 112], np.uint8)
        for i in range(112):
            for j in range(112):

                if output_v[i,j]>0.7:
                    output[i, j] = 255
                else:
                    output[i,j]= 0

        # print(tf.shape(output_v))
        # print(output_v)
        cv2.imwrite('E:\code\Mytry\data\output\Tiszadob'+str(step)+'.bmp',output)
        # cv2.imshow('output',output_v)
        # cv2.imshow('output2', output)
        # scipy.misc.imsave('E:\code\Mytry\data\output\Tiszadob'+str(step)+'.bmp',output_v)
        # print(tf.shape(y_v))
        # tensor1 = tf.squeeze(y_v)
        # image_tensor = tf.placeholder(tf.float32, [112, 112, 1])
        # image_tensor = tf.expand_dims(tensor1, -1)
        # print(tf.shape(image_tensor))
        # cv2.imshow('y', image_tensor.eval() )
        # cv2.waitKey(0)
#     embed = siamese.o1.eval({siamese.x1: mnist.test.images})
#     embed.tofile('embed.txt')
#
# # visualize result
# x_test = mnist.test.images.reshape([-1, 28, 28])
# visualize.visualize(embed, x_test)

