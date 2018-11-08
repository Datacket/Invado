import numpy as np
import tensorflow as tf
import pandas as pd
import random
import matplotlib.pyplot as plt

class DatasetSplit(object):
    def __init__(self, x, y, bs):
        self.x = x
        self.y = y
        self.bs = bs
        self.start_split = 0
        self.its = 1
        
    def get_next_batch(self):
        flag = True
        while flag:
            self.start_split = self.its * self.bs
            ee = min(len(self.x), self.start_split + self.bs)
            self.its += 1
            if len(self.x) == ee:
                flag = False
            yield self.x[self.start_split : ee], self.y[self.start_split : ee]
        
class Dataset(object):
    x = None
    y = None
    def __init__(self, X, Y, bs, split=[0.9, 0.1]):
        self.x = X
        self.y = Y
        self.bs = bs
        
        self.train_x = self.x[: int(0.9 * len(self.x))]
        self.train_y = self.y[: int(0.9 * len(self.y))]
        
        self.test_x = self.x[int(0.9 * len(self.x)):]
        self.test_y = self.y[int(0.9 * len(self.y)):]
        
    def train(self):
        return DatasetSplit(self.train_x, self.train_y, bs=self.bs)
    
    def test(self):
        return DatasetSplit(self.test_x, self.test_y, bs=len(self.test_x))

class WildlifeCraziness(object):

	def __init__(self, n_in, n_out, hid1=1024, hid2=512, learning_rate=0.001, n_epochs=10, batch_size=64, interval=5):
		self.n_in = n_in
		self.n_out = n_out
		self.hid1 = hid1
		self.hid2 = hid2
		self.lr = learning_rate
		self.n_epochs = n_epochs
		self.bs = batch_size
		self.interval = interval
		# Split = [0.9, 0.1]
		self.total_samples = 9000
	
	def fit(self):
		self.x = tf.placeholder(tf.float32, shape=(None, self.n_in))
		self.y = tf.placeholder(tf.float32, shape=(None, self.n_out))

		# dense1
		self.W1 = tf.get_variable('weights1', shape=(self.n_in, self.hid1))
		self.b1 = tf.get_variable('bias1', shape=(self.hid1))
		self.dense1 = tf.nn.relu(tf.matmul(self.x, self.W1, name='matmul2') + self.b1)

		# dense2
		self.W2 = tf.get_variable('weights2', shape=(self.hid1, self.hid2))
		self.b2 = tf.get_variable('bias2', shape=(self.hid2))
		self.dense2 = tf.nn.relu(tf.matmul(self.dense1, self.W2, name='matmul2') + self.b2, name='relu2')

		# dense2
		self.W3 = tf.get_variable('weights3', shape=(self.hid2, self.n_out))
		self.b3 = tf.get_variable('bias3', shape=(self.n_out))
		self.dense3 = tf.matmul(self.dense2, self.W3, name='matmul3') + self.b3
#		print(self.dense3.shape)
		self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.dense3, name='softmax1')
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam1')
		self.optimizer = self.optimizer.minimize(self.loss)
		self.probs=tf.nn.softmax(self.dense3)	
		self.sess = tf.Session()

		self.sess.run(tf.global_variables_initializer())
		for epoch in range(self.n_epochs):
			dataset_train = self.dataset.train().get_next_batch()
			for batch in range((self.total_samples // self.bs)):
				xs, ys = next(dataset_train)
				_, curr_loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.x:xs, self.y:ys})
				print("Epoch #{}: {}".format(epoch, curr_loss))
			'''
				if epoch % self.interval == 0:
				loss = 'NO VALID DATASET'
				print ('Loss for Epoch ', epoch , ' is ', loss)
			'''

	def predict(self, test_x,one_hot_animals):
		'''
		df = pd.read_csv(path)
		y = np.array(df.iloc[:, -1])
		x = np.array(df.iloc[:, :-1])
		dataset = Dataset(x, y, batch_size)
		'''
		
		# Predicting on test data
		probs = self.sess.run(self.probs, feed_dict={self.x:test_x})
		probs=probs.flatten().tolist()
		#print(probs)
		#print (np.array(probs).shape)
		d={}
		l=list(enumerate(probs))
		l=sorted(l,reverse=True,key=lambda x:x[1])
		d={one_hot_animals[i]:x for i,x in l}
		return d

	def close_session(self):
		self.sess.close()
		print ('Session Closed Successfully!')


	def load_dataset(self, x, y):
		self.dataset = Dataset(x, y, self.bs)


# Dry run
'''
n_in = 25
n_out = 10
hid1 = 1024
hid2 = 512
learning_rate = 0.001
n_epochs = 1000
batch_size = 128
path = './data.csv'''
