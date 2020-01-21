import tensorflow as tf
import numpy as np

class BinCNN:
    def __init__(self, model_path):
        self.model = np.load(model_path)
        print("Printing files in the model:")
        for x in self.model.files:
            print(x + " " + str(self.model[x].shape))

    def mod_layer(self, inp, val):
        res = tf.floormod(inp, val)
        return res

    #Implements y = +1 for x > 0 and y = -1 for x <= 0
    def sign_binarize(self, inp):
        h_sig = tf.clip_by_value((inp+1.)/2., 0, 1)
        round_out = tf.round(h_sig)
        round_fin = h_sig + tf.stop_gradient(round_out - h_sig)
        return (2.*round_fin - 1.)

    def bin_dense_layer(self, in_act, in_w, in_b, name='bin_dense_l'):
        bin_w = self.sign_binarize(in_w)
        res = tf.matmul(in_act, bin_w)
        res = tf.nn.bias_add(res, in_b) #tf.cast(in_b, dtype=tf.int32)        
        return res 

    def bin_conv2d(self, in_act, in_w, in_b, padding, name='bin_conv2d_l'):
        bin_w = self.sign_binarize(in_w)
        res = tf.nn.conv2d(in_act, bin_w, strides=[1, 1, 1, 1], padding=padding, name=name, data_format='NHWC')
        res = tf.nn.bias_add(res, in_b, data_format='NHWC')
        return res

    def compute_psi_phi(self, lname, epsilon):
        _gamma = self.model[lname+'_gamma']
        _beta = self.model[lname+'_beta']
        _mean = self.model[lname+'_mean']
        _variance = self.model[lname+'_variance']
        _psi = _gamma / tf.sqrt(_variance+epsilon)
        _phi = _beta - _psi*_mean
        return _psi, _phi

    def build_convnet(self, in_act):
        epsilon = 1e-4
        modk = 1024
        _psi, _phi = self.compute_psi_phi('l0', epsilon=epsilon)
        l0 = self.bin_conv2d(in_act, self.model['l0_w'], tf.math.round(self.model['l0_b']+_phi/_psi), padding = 'SAME', name='bin_conv2d_l0')
        l0 = self.mod_layer(l0, modk)
        #l0 = _psi * l0 + _phi
        #l0 = tf.nn.batch_normalization(tf.cast(l0, dtype=tf.float32), mean=self.model['l0_mean'], variance=self.model['l0_variance'], offset=self.model['l0_beta'], scale=self.model['l0_gamma'], variance_epsilon=1e-4, name='bin_conv2d0_bn')
        #l0 = self.sign_binarize(l0)
        l0 = -self.sign_binarize(l0 - modk/2.)

        _psi, _phi = self.compute_psi_phi('l1', epsilon=epsilon)
        l1 = self.bin_conv2d(l0, self.model['l1_w'], tf.math.round(self.model['l1_b']+_phi/_psi), padding = 'SAME', name='bin_conv2d_l1')
        l1 = self.mod_layer(l1, modk)
        #l1 = _psi * l1 + _phi
        #l1 = tf.nn.batch_normalization(tf.cast(l1, dtype=tf.float32), mean=self.model['l1_mean'], variance=self.model['l1_variance'], offset=self.model['l1_beta'], scale=self.model['l1_gamma'], variance_epsilon=1e-4, name='bin_conv2d1_bn')
        l1 = -self.sign_binarize(l1 - modk/2.)
        l1 = tf.nn.max_pool(l1, ksize=(2, 2), strides=(2, 2), padding='VALID')
        #l1 = self.sign_binarize(l1)

        _psi, _phi = self.compute_psi_phi('l2', epsilon=epsilon)
        l2 = self.bin_conv2d(l1, self.model['l2_w'], tf.math.round(self.model['l2_b']+_phi/_psi), padding = 'SAME', name='bin_conv2d_l2')
        l2 = self.mod_layer(l2, modk)
        #l2 = tf.nn.batch_normalization(tf.cast(l2, dtype=tf.float32), mean=self.model['l2_mean'], variance=self.model['l2_variance'], offset=self.model['l2_beta'], scale=self.model['l2_gamma'], variance_epsilon=1e-4, name='bin_conv2d2_bn')
        l2 = -self.sign_binarize(l2 - modk/2.)
        #l2 = self.sign_binarize(l2)
    
        _psi, _phi = self.compute_psi_phi('l3', epsilon=epsilon)
        l3 = self.bin_conv2d(l2, self.model['l3_w'], tf.math.round(self.model['l3_b']+_phi/_psi), padding = 'SAME', name='bin_conv2d_l3')
        l3 = self.mod_layer(l3, modk)
        #l3 = tf.nn.batch_normalization(tf.cast(l3, dtype=tf.float32), mean=self.model['l3_mean'], variance=self.model['l3_variance'], offset=self.model['l3_beta'], scale=self.model['l3_gamma'], variance_epsilon=1e-4, name='bin_conv2d3_bn')
        l3 = -self.sign_binarize(l3 - modk/2.)
        l3 = tf.nn.max_pool(l3, ksize=(2, 2), strides=(2, 2), padding='VALID')
        #l3 = self.sign_binarize(l3)

        _psi, _phi = self.compute_psi_phi('l4', epsilon=epsilon)
        l4 = self.bin_conv2d(l3, self.model['l4_w'], tf.math.round(self.model['l4_b']+_phi/_psi), padding = 'SAME', name='bin_conv2d_l4')
        l4 = self.mod_layer(l4, modk)
        #l4 = tf.nn.batch_normalization(tf.cast(l4, dtype=tf.float32), mean=self.model['l4_mean'], variance=self.model['l4_variance'], offset=self.model['l4_beta'], scale=self.model['l4_gamma'], variance_epsilon=1e-4, name='bin_conv2d4_bn')
        l4 = -self.sign_binarize(l4 - modk/2.)
        #l4 = self.sign_binarize(l4)

        _psi, _phi = self.compute_psi_phi('l5', epsilon=epsilon)
        l5 = self.bin_conv2d(l4, self.model['l5_w'], tf.math.round(self.model['l5_b']+_phi/_psi), padding = 'SAME', name='bin_conv2d_l5')
        l5 = self.mod_layer(l5, modk)
        #l5 = tf.nn.batch_normalization(tf.cast(l5, dtype=tf.float32), mean=self.model['l5_mean'], variance=self.model['l5_variance'], offset=self.model['l5_beta'], scale=self.model['l5_gamma'], variance_epsilon=1e-4, name='bin_conv2d5_bn')
        l5 = -self.sign_binarize(l5 - modk/2.)
        l5 = tf.nn.max_pool(l5, ksize=(2, 2), strides=(2, 2), padding='VALID')
        #l5 = self.sign_binarize(l5)

        l5 = tf.reshape(l5, [-1, l5.shape[1]*l5.shape[2]*l5.shape[3]])

        _psi, _phi = self.compute_psi_phi('l6', epsilon=epsilon)
        l6 = self.bin_dense_layer(l5, self.model['l6_w'], tf.math.round(self.model['l6_b']+_phi/_psi), name='dense0')
        l6 = self.mod_layer(l6, modk)
        #l6 = tf.nn.batch_normalization(tf.cast(l6, dtype=tf.float32), mean=self.model['l6_mean'], variance=self.model['l6_variance'], offset=self.model['l6_beta'], scale=self.model['l6_gamma'], variance_epsilon=1e-4, name='bin_conv2d6_bn')
        l6 = -self.sign_binarize(l6 - modk/2.)
        #l6 = self.sign_binarize(l6)

        _psi, _phi = self.compute_psi_phi('l7', epsilon=epsilon)
        l7 = self.bin_dense_layer(l6, self.model['l7_w'], tf.math.round(self.model['l7_b']+_phi/_psi), name='dense1')
        l7 = self.mod_layer(l7, modk)
        #l7 = tf.nn.batch_normalization(tf.cast(l7, dtype=tf.float32), mean=self.model['l7_mean'], variance=self.model['l7_variance'], offset=self.model['l7_beta'], scale=self.model['l7_gamma'], variance_epsilon=1e-4, name='bin_conv2d7_bn')
        l7 = -self.sign_binarize(l7 - modk/2.)
        #l7 = self.sign_binarize(l7)

        _psi, _phi = self.compute_psi_phi('l8', epsilon=epsilon)
        l8 = self.bin_dense_layer(l7, self.model['l8_w'], tf.math.round(self.model['l8_b']+_phi/_psi), name='dense2')
        #l8 = tf.nn.batch_normalization(tf.cast(l8, dtype=tf.float32), mean=self.model['l8_mean'], variance=self.model['l8_variance'], offset=self.model['l8_beta'], scale=self.model['l8_gamma'], variance_epsilon=1e-4, name='bin_conv2d8_bn')

        return l8

