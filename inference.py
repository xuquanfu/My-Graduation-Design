import tensorflow as tf
import numpy as np


class siamese:

    # Create model
    def __init__(self):
        #self.x1 = tf.placeholder(tf.float32, [None, 784])
        #self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.x1 = tf.placeholder(tf.float32, [None, 112, 112, 3])
        self.x2 = tf.placeholder(tf.float32, [None, 112, 112, 3])

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        # Create loss
        #self.y_=tf.placeholder(tf.float32, [None])
        self.y_ = tf.placeholder(tf.float32, [None,112, 112])
        self.output=tf.placeholder(tf.float32, [112,112,1])
        self.loss = self.my_loss()

    def network(self, x):
        # 还要加归一化层bn层
        #bx=tf.reshape(x,[-1,28,28,1])

        l1 = self.mycnn_layer(x,[3,3,3,64],[1,1,1,1], "l1")
        active1=tf.nn.relu(l1)
        l2 = self.mycnn_layer(active1,[3,3,64,64],[1,1,1,1], "l2")
        active2 = tf.nn.relu(l2)
        l3 = self.mycnn_layer(active2,[5,5,64,64],[1,1,1,1], "l3")
        active3 = tf.nn.relu(l3)
        l4 = self.mycnn_layer(active3, [5, 5, 64, 32], [1, 1, 1, 1], "l4")
        active4 = tf.nn.relu(l4)
        l5 = self.mycnn_layer(active4, [1, 1, 32, 16], [1, 1, 1, 1], "l5")
        active5 = tf.nn.relu(l5)
        l5_out=tf.nn.l2_normalize(active5,dim=3)
        return l5_out


    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2

        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc




    def mycnn_layer(self, bottom, input,stride, myname):
        kernel = tf.get_variable(name=myname+'W',shape=input, initializer=tf.random_normal_initializer(mean=0, stddev=1))
        biases = tf.get_variable(name=myname+'b',shape=np.shape(kernel)[3], initializer=tf.random_normal_initializer(mean=0, stddev=1))
        return tf.nn.bias_add(tf.nn.conv2d(bottom, kernel, stride, padding='SAME'), biases)


    def my_loss(self):

        # distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.o1, self.o2)),axis=3))
        distance = tf.square(tf.subtract(self.o1, self.o2))
        # print(tf.argmax(Subtract,axis=3))
        # print(tf.reduce_max(Subtract, axis=3))

        # softmax=tf.nn.softmax(logits=distance, dim=-1)
        Normalize=tf.nn.l2_normalize(distance,dim=3)
        # tensor=tf.reduce_max(Normalize, axis=3)
        # tensor1 = tf.squeeze(tensor)
        # image_tensor = tf.expand_dims(tensor1, -1)
        # self.output=image_tensor
        #
        Noisy_or=tf.subtract(1.0,tf.reduce_prod(tf.subtract(1.0,Normalize),axis=3))
        tensor1 = tf.squeeze(Noisy_or)
        image_tensor = tf.expand_dims(tensor1, -1)
        # print(Noisy_or)
        self.output = image_tensor
        changeloss=9.0*tf.reduce_sum(tf.multiply(self.y_,tf.subtract(1.0,Noisy_or)))
        unchangeloss=0.7340*tf.reduce_sum(tf.multiply(tf.subtract(1.0,self.y_),Noisy_or))
        # print(softmax)
        # ll=tf.reduce_max(softmax, axis=3)
        # print(ll)
        # self.output=tf.reshape(distance,[112,112,1])
        #
        # unchange_w = tf.subtract(1.0, self.y_)
        # unchangeloss = tf.reduce_sum(tf.multiply(unchange_w,distance))
        #m = tf.constant(0.5, dtype=None, shape=(1,112,112), name='m')
        # print(m)
        # changeloss =  tf.reduce_sum(tf.multiply(self.y_,tf.maximum(0.0,tf.subtract(0.5,distance))))
        # changeloss = tf.reduce_sum(tf.multiply(self.y_,  tf.subtract(1.0, tf.reduce_max(softmax, axis=3))))
        # changeloss = tf.reduce_sum(tf.multiply(self.y_, tf.subtract(1.0, tf.reduce_max(Normalize, axis=3))))
        loss=unchangeloss+changeloss

        # se=tf.Session()
        # for i in range(112):
        #     for j in range (112):
        #         # print(self.y_[0,i,j])
        #         # print((Suntract[0,i,j])[tf.argmax(Suntract[0,i,j])])
        #         # loss=loss+(1-self.y_[0,i,j])*np.max(Suntract[0,i,j])-self.y_[0,i,j]*np.max(Suntract[0,i,j])
        #         loss=loss+(1-self.y_[0,i,j])*(Suntract[0,i,j])[tf.argmax(Suntract[0,i,j])]-self.y_[0,i,j]*(Suntract[0,i,j])[tf.argmax(Suntract[0,i,j])]
        #     print(loss)
                # print(se.run(np.max(Suntract[0,i,j]),np.min(Suntract[0,i,j])))



        #se.close()
        # print(Suntract)
        # return tf.reduce_sum(Suntract)
        return loss








    def loss_with_spring(self):
        margin = 10.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
