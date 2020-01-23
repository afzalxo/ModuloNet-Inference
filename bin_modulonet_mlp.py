import tensorflow as tf
import numpy as np

class BinModMLP:
    #Load pretrained npz model
    def __init__(self, model_path):
        self.model = np.load(model_path)
        for x in self.model.files:
            print(x + " " + str(self.model[x].shape))

    #Modulo operation wrapper
    def mod_layer(self, inp, val):
        res = tf.floormod(tf.cast(inp, dtype=tf.int32), val)
        return tf.cast(res, dtype=tf.int16)

    #Activation function y = sign(x); Implements y = +1 for x > 0 and y = -1 for x <= 0
    def sign_binarize(self, inp):
        h_sig = tf.clip_by_value((tf.cast(inp, dtype=tf.float32)+1.)/2., 0, 1)
        round_out = tf.round(h_sig)
        round_fin = h_sig + tf.stop_gradient(round_out - h_sig)
        return tf.cast(2.*round_fin - 1., dtype=tf.int32)

    #Implements y = x * w + b where w are assumed to be pre-binarized
    def bin_dense_layer(self, in_act, in_w, in_b, name='bin_dense_l'):
        #Input weights are assumed to be pre-binarized
        bin_w = in_w
        bin_w = tf.cast(bin_w, dtype=tf.int32)
        res = tf.matmul(tf.cast(in_act, dtype=tf.int32), bin_w)
        res = tf.cast(res, dtype=tf.int16)
        res = tf.nn.bias_add(res, in_b) 
        return res 
    
    #Computes psi and phi values. Entirely dependant on learned BN params, hence will be precomputed in hardware. This fn is unused in modulonet since b_new has been precomputed
    def compute_psi_phi(self, lname, epsilon):
        _gamma = self.model[lname+'_gamma']
        _beta = self.model[lname+'_beta']
        _mean = self.model[lname+'_mean']
        _variance = self.model[lname+'_variance']
        _psi = _gamma / tf.sqrt(_variance+epsilon)
        _phi = _beta - _psi*_mean
        return _psi, _phi

    #Precomputation of bin_w and b_new to store them in a differnet npz for modulonet inference
    def _precompute(self):
        w_vec = []
        b_new_vec = []
        for i in range(0, 4):
            w_str = 'l'+str(i)
            new_w = self.sign_binarize(self.model[w_str+'_w'])
            psi, phi = self.compute_psi_phi(w_str, 1e-4)
            new_b = tf.math.round(self.model[w_str+'_b'] + phi/psi)
            if i == 0:
                new_b = 127*new_b
            w_vec.append(new_w)
            b_new_vec.append(new_b)
        return w_vec, b_new_vec

    #Build the TF graph for ModuloNet
    def build(self, in_act):
        epsilon = 1e-4
        k0 = 32768
        k1 = 1024
        k2 = 1024
        k3 = 4096
        in_act = tf.contrib.layers.flatten(in_act)
        #Layer 0 starts here
        #Restoring b_new from the pretrained model file. b_new = round(b_old + phi/psi). Rounding leads to some loss of accuracy
        biases0 = tf.cast(self.model['l0_bnew'], dtype=tf.int16)
        #Dense layer implements y = w*x + b_new. Please note self.model['lx_w'] are already binarized.
        layer0_dense = self.bin_dense_layer(in_act, self.model['l0_w'], biases0, name='layer0_dense')
        #Apply mod k to the dense layer output. In hardware, modulo would be applied after every multiplication of the elements of matrix bin(w) with the matrix x. This is mathematically equivalent to the current software implementation as below due to associativity of modular additions.  
        layer0_dense = self.mod_layer(layer0_dense, k0)
        #Applying shifted, scaled sign binarization.
        layer0_sig = -self.sign_binarize(layer0_dense - tf.cast(k0/2, dtype=tf.int16))

        #Layer 1 starts here
        #_psi1, _phi1 = self.compute_psi_phi('l1', epsilon)
        #biases1 = tf.cast(tf.math.round(self.model['l1_b'] +_phi1/(_psi1)), dtype=tf.int16)
        biases1 = tf.cast(self.model['l1_bnew'], dtype=tf.int16)
        layer1_dense = self.bin_dense_layer(layer0_sig, self.model['l1_w'], biases1, name='layer1_dense')
        layer1_dense = self.mod_layer(layer1_dense, k1)
        layer1_sig = -self.sign_binarize(layer1_dense - tf.cast(k1/2, dtype=tf.int16))

        #Layer 2 starts here
        #_psi2, _phi2 = self.compute_psi_phi('l2', epsilon)
        #biases2 = tf.cast(tf.math.round(self.model['l2_b'] +_phi2/(_psi2)), dtype=tf.int16)
        biases2 = tf.cast(self.model['l2_bnew'], dtype=tf.int16)
        layer2_dense = self.bin_dense_layer(layer1_sig, self.model['l2_w'], biases2, name='layer2_dense')
        layer2_dense = self.mod_layer(layer2_dense, k2)
        layer2_sig = -self.sign_binarize(layer2_dense-tf.cast(k2/2, dtype=tf.int16))

        #Layer 3 starts here
        #_psi3, _phi3 = self.compute_psi_phi('l3', epsilon)
        #biases3 = tf.cast(tf.math.round(self.model['l3_b'] + _phi3/_psi3), dtype=tf.int16)
        biases3 = tf.cast(self.model['l3_bnew'], dtype=tf.int16)
        layer3_dense = self.bin_dense_layer(layer2_sig, self.model['l3_w'], biases3, name='layer3_dense')
        layer3_out = self.mod_layer(layer3_dense, k3)
        layer3_out = tf.cast(layer3_out, dtype=tf.int16)
        return layer3_out 
