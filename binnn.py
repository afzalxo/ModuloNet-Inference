import tensorflow as tf
import numpy as np

class BinNN:
    def __init__(self, model_path):
        self.model = np.load(model_path)
        self.frac_nbits = 0
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

    def bin_dense_layer(self, in_act, in_w, in_b, name='Bin_Dense_L'):
        bin_w = self.sign_binarize(in_w)
        res = tf.matmul(in_act, bin_w)
        res = tf.nn.bias_add(res, in_b) #tf.cast(in_b, dtype=tf.int32))#self.sign_binarize(in_b), dtype=tf.int32))        
        return res 

    def bin_conv2d(self, in_act, in_w, in_b, name='bin_conv2d_l'):
        bin_w = self.sign_binarize(in_w)
        res = tf.nn.conv2d(in_act, bin_w, strides=[1, 1, 1, 1], padding=padding, name=name, data_format='NHWC')
        res = tf.nn.bias_add(res, in_b)
        return res

    def bn_wrapper(self, x, mean, variance, offset, scale, epsilon, mod_k):
        y = tf.where(tf.greater(x, mod_k/2.), tf.nn.batch_normalization(tf.cast(x, dtype=tf.float32), mean=mean+mod_k, variance=variance, offset=offset, scale = scale, variance_epsilon=epsilon, name='bn_modified'), tf.nn.batch_normalization(tf.cast(x, dtype=tf.float32), mean=mean, variance=variance, offset = offset, scale=scale, variance_epsilon=epsilon, name='bn_normal'))
        return y

    def modified_bn2(self, x, scale, epsilon):
        _mean, _variance = tf.nn.moments(x, [0])
        phi = scale / tf.sqrt(_variance + epsilon)
        return phi*x, phi

    def compute_psi_phi(self, act, lname, epsilon):
        _gamma = self.model[lname+'_gamma']
        _beta = self.model[lname+'_beta']
        _mean, _variance = tf.nn.moments(act, [0])
        _psi = _gamma / tf.sqrt(_variance+epsilon)
        _phi = _beta - _psi*_mean
        return _psi, _phi

    def compute_psi_phi0(self, lname, epsilon):
        _gamma = self.model[lname+'_gamma']
        _beta = self.model[lname+'_beta']
        _mean = self.model[lname+'_mean']
        _variance = self.model[lname+'_variance']
        _psi = _gamma / tf.sqrt(_variance+epsilon)
        _phi = _beta - _psi*_mean
        return _psi, _phi

    def build(self, in_act):
        epsilon = 1e-4
        k0 = 32768.
        k1 = 1024.
        k2 = 1024.
        in_act = tf.contrib.layers.flatten(in_act)
        #in_act = self.sign_binarize(in_act)
        _psi0, _phi0 = self.compute_psi_phi0('l0', epsilon)
        biases0 = tf.cast(tf.cast(127*tf.math.round(self.model['l0_b'] +_phi0/(_psi0)), dtype=tf.int16), dtype=tf.float32)
        layer0_dense = self.bin_dense_layer(in_act, self.model['l0_w'], biases0, name='layer0_dense')
        layer0_dense = self.mod_layer(layer0_dense, k0)
        layer0_bn = layer0_dense
        #layer0_bn, mean_psi0 = self.modified_bn2(tf.cast(layer0_dense, dtype=tf.float32), scale=self.model['l0_gamma'], epsilon=1e-4)
        #layer0_bn = self.bn_wrapper(tf.cast(layer0_dense, dtype=tf.float32), mean=self.model['l0_mean'], variance=self.model['l0_variance'], offset=self.model['l0_beta'], scale=self.model['l0_gamma'], epsilon=1e-4, mod_k=4500.)
        #mean, variance = tf.nn.moments(layer0_dense,[0])
        #layer0_bn = tf.nn.batch_normalization(tf.cast(layer0_dense, dtype=tf.float32), mean=mean, variance=variance, offset=self.model['l0_beta'], scale=self.model['l0_gamma'], variance_epsilon=1e-4, name='layer0_bn')
        #--Baseline---
        #_psi0, _phi0 = self.compute_psi_phi(layer0_bn, 'l0', epsilon)
        #layer0_bn = _psi0*layer0_bn + _phi0
        #layer0_sig = self.sign_binarize(layer0_bn)
        #--Baseline---
        #case1 = tf.where(tf.greater_equal(_phi0, 0.), (-k/2.)+(_phi0/_psi0), tf.ones_like(_phi0)*(-k/2.))
        #case1 = layer0_bn*_psi0+_phi0-(tf.reduce_mean(_psi0)*k/2. + tf.reduce_mean(_phi0))
        #layer0_sig = -self.sign_binarize(case1)#layer0_bn + case1)
        layer0_sig = -self.sign_binarize(layer0_dense - k0/2.)
        #layer0_sig = self.sign_binarize(layer0_dense)

        _psi1, _phi1 = self.compute_psi_phi0('l1', epsilon)
        biases1 = tf.cast(tf.cast(tf.math.round(self.model['l1_b'] +_phi1/(_psi1)), dtype=tf.int16), dtype=tf.float32)
        layer1_dense = self.bin_dense_layer(layer0_sig, self.model['l1_w'], biases1, name='layer1_dense')
        layer1_dense = self.mod_layer(layer1_dense, k1)
        layer1_bn = layer1_dense
        #layer1_bn, mean_psi1 = self.modified_bn2(tf.cast(layer1_dense, dtype=tf.float32), scale=self.model['l1_gamma'], epsilon=1e-4)
        #layer1_bn = self.bn_wrapper(tf.cast(layer1_dense, dtype=tf.float32), mean=self.model['l1_mean'], variance=self.model['l1_variance'], offset=self.model['l1_beta'], scale=self.model['l1_gamma'], epsilon=1e-4, mod_k=4500.)
        #mean, variance = tf.nn.moments(layer1_dense,[0])
        #layer1_bn = tf.nn.batch_normalization(tf.cast(layer1_dense, dtype=tf.float32), mean=mean, variance=variance, offset=self.model['l1_beta'], scale=self.model['l1_gamma'], variance_epsilon=1e-4, name='layer1_bn')
        #_psi1, _phi1 = self.compute_psi_phi(layer1_bn, 'l1', epsilon)
        #--Baseline---
        #layer1_bn = _psi1*layer1_bn + _phi1
        #layer1_sig = self.sign_binarize(layer1_bn)
        #--Baseline---
        #case1 = tf.where(tf.greater_equal(_phi1, 0.), (-k/2.)+(_phi1/_psi1), tf.ones_like(_phi1)*(-k/2.))
        #layer1_sig = -self.sign_binarize(layer1_bn - k/2. + _phi1_cond/_psi1)#- k*mean_psi1/2.)
        #case1 = layer1_bn*_psi1+_phi1-(tf.reduce_mean(_psi1)*k/2. + tf.reduce_mean(_phi1))
        #layer1_sig = -self.sign_binarize(case1)#layer1_bn + case1)
        #layer1_sig = -self.sign_binarize(layer1_dense - k/2.)
        layer1_sig = -self.sign_binarize(layer1_dense - k1/2.)

        _psi2, _phi2 = self.compute_psi_phi0('l2', epsilon)
        biases2 = tf.cast(tf.cast(tf.math.round(self.model['l2_b'] +_phi2/(_psi2)), dtype=tf.int16), dtype=tf.float32)
        layer2_dense = self.bin_dense_layer(layer1_sig, self.model['l2_w'], biases2, name='layer2_dense')
        layer2_dense = self.mod_layer(layer2_dense, k2)
        layer2_bn = layer2_dense
        #layer2_bn, mean_psi2 = self.modified_bn2(tf.cast(layer2_dense, dtype=tf.float32), scale=self.model['l2_gamma'], epsilon=1e-4)
        #layer2_bn = self.bn_wrapper(tf.cast(layer2_dense, dtype=tf.float32), mean=self.model['l2_mean'], variance=self.model['l2_variance'], offset=self.model['l2_beta'], scale=self.model['l2_gamma'], epsilon=1e-4, mod_k=4500.)
        #mean, variance = tf.nn.moments(layer2_dense,[0])
        #layer2_bn = tf.nn.batch_normalization(tf.cast(layer2_dense, dtype=tf.float32), mean=mean, variance=variance, offset=self.model['l2_beta'], scale=self.model['l2_gamma'], variance_epsilon=1e-4, name='layer2_bn')
        #_psi2, _phi2 = self.compute_psi_phi(layer2_bn, 'l2', epsilon)
        #--Baseline---
        #layer2_bn = _psi2*layer2_bn + _phi2
        #layer2_sig = self.sign_binarize(layer2_bn)
        #-------------
        #case1 = tf.where(tf.greater_equal(_phi2, 0.), (-k/2.)+(_phi2/_psi2), tf.ones_like(_phi2)*(-k/2.))
        #layer2_sig = -self.sign_binarize(layer2_bn - k/2.  + _phi2_cond/_psi2)#- k*mean_psi2/2.)
        #case1 = layer2_bn*_psi2+_phi2-(tf.reduce_mean(_psi2)*k/2. + tf.reduce_mean(_phi2))
        #layer2_sig = -self.sign_binarize(case1)#layer2_bn + case1)
        #layer2_sig = -self.sign_binarize(layer2_dense - k/2.)
        layer2_sig = -self.sign_binarize(layer2_dense-k2/2.)

        layer3_dense = self.bin_dense_layer(layer2_sig, self.model['l3_w'], self.model['l3_b'], name='layer3_dense')
        #layer3_dense = self.mod_layer(layer3_dense, k)
        #layer3_bn, mean_psi = self.modified_bn2(tf.cast(layer3_dense, dtype=tf.float32), scale=self.model['l3_gamma'], epsilon=1e-4)
        #layer3_bn = self.bn_wrapper(tf.cast(layer3_dense, dtype=tf.float32), mean=self.model['l3_mean'], variance=self.model['l3_variance'], offset=self.model['l3_beta'], scale=self.model['l3_gamma'], epsilon=1e-4, mod_k=4500.)
        mean, variance = tf.nn.moments(layer3_dense,[0])
        layer3_bn = tf.nn.batch_normalization(tf.cast(layer3_dense,dtype=tf.float32), mean=mean, variance=variance, offset=self.model['l3_beta'], scale=self.model['l3_gamma'], variance_epsilon=1e-4, name='layer3_bn')
        layer3_out = layer3_bn#self.mod_layer(layer3_bn, 8.)
        #layer3_bn = tf.cast(layer3_dense, dtype=tf.float32)
        return layer3_out, layer3_bn, layer2_dense, layer2_bn, _psi0, _phi0, _psi1, _phi1, _psi2, _phi2, biases0, biases1, biases2

    def build_convnet(self, in_act):


'''
inp = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[-2, 0, -8, -9, 3, 9, -2, 9, 0.25, 0.75, -0.25, -0.75]], dtype=np.float32)
inp1 = tf.placeholder(dtype=tf.float32, shape=(2, 12))
res = bin_dense_layer(inp1, 2048)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ret=sess.run(res, feed_dict={inp1:inp})
    print(ret)
'''
