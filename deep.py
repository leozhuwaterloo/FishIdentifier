import _pickle as cPickle
import tensorflow as tf
import json
import numpy as np
from PIL import Image

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def next_batch(num, images, labels):
    idx = np.arange(0 , len(images))
    np.random.shuffle(idx)
    idx = idx[:num]
    image_shuffle = [images[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(image_shuffle), np.asarray(labels_shuffle)


class NeutralNetwork(object):
    def __init__(self, id_count, fish_group_id):
        self.fish_group_id = fish_group_id
        self.x = tf.placeholder(tf.float32, [None, 96*64*3], name='x-input')
        self.y_ = tf.placeholder(tf.float32, [None, id_count], name='y-input')
        self.keep_prob = tf.placeholder(tf.float32)

        x_image = tf.reshape(self.x, [-1, 96, 64, 3])

        h_conv1 = self.deep_nn_layer(x_image, [3, 3, 3, 16], [16], tf.nn.relu, conv2d)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = self.deep_nn_layer(h_pool1, [5, 5, 16, 32], [32], tf.nn.relu, conv2d)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_drop = tf.nn.dropout(h_pool2, self.keep_prob)

        h_conv3 = self.deep_nn_layer(h_pool2_drop, [5, 5, 32, 64], [64], tf.nn.relu, conv2d)
        h_pool3 = max_pool_2x2(h_conv3)

        h_pool3_flat = tf.reshape(h_pool3, [-1, 12 * 8 * 64])

        h_fc = self.deep_nn_layer(h_pool3_flat, [12 * 8 * 64, 512], [512], tf.nn.relu, tf.matmul)
        h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

        self.y_conv = self.deep_nn_layer(h_fc_drop, [512, id_count], [id_count], tf.identity, tf.matmul)

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y_conv))


        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def deep_nn_layer(self, input_tensor, weight_dim, bias_dim, act, handle):
        weights = self.weight_variable(weight_dim)
        biases = self.bias_variable(bias_dim)
        return act(handle(input_tensor, weights) + biases)

    def train(self, training_image, training_label, attempt_n, batch_size, learning_rate, test_image, test_label):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess, "Fish" + str(self.fish_group_id) + "/fish_conv.ckpt")

        for i in range(attempt_n):
            batch_xs, batch_ys = next_batch(batch_size, training_image, training_label)
            if i % 100 == 99:
                print("Step %d: %f" % (i, sess.run(self.accuracy,
                                                   feed_dict={self.x: test_image, self.y_: test_label,
                                                              self.keep_prob: 1.0})))
                saver.save(sess, "Fish" + str(self.fish_group_id) + "/fish_conv.ckpt")
            sess.run(train_step,
                     feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5})
        sess.close()

    def predict(self, path, test_img):
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, path)

        result = tf.argmax(self.y_conv, 1)
        res = sess.run(result,
                                      feed_dict={self.x: test_img, self.keep_prob: 1.0})
        sess.close()
        print(res[0])
        return res[0]


def load_map(mapfile):
    with open(mapfile, 'r') as mfile:
        fish_map = json.load(mfile)

    return fish_map

def load_data(datafile, mapfile):
    with open(datafile, 'rb') as pfile:
        data= cPickle.load(pfile)

    return data, load_map(mapfile)

if __name__ == "__main__":
    fish_group = 1
    data, fish_map = load_data("data"+ str(fish_group) + ".p", "fishMap" + str(fish_group) + ".json")
    
    max_id = fish_map["max_id"]
    min_id = fish_map["min_id"]

    print(len(data[0]))
    print(len(data[2]))
    print(min_id)
    print(max_id)
    nn = NeutralNetwork(max_id - min_id + 1, fish_group)
    nn.train(data[0], data[1], 10000, 10, 1e-3, data[2], data[3])



