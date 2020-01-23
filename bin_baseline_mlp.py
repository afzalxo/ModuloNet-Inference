import tensorflow as tf
import numpy as np

class BinBaseMLP:
    #Load pretrained npz model
    def __init__(self, model_path):
        self.model = np.load(model_path)
        for x in self.model.files:
            print(x + " " + str(self.model[x].shape))

    #Modulo operation wrapper
    def mod_layer(self, inp, val):
        res = tf.floormod(inp, val)
        return res

    #Activation function y = sign(x); Implements y = +1 for x > 0 and y = -1 for x <= 0
    def sign_binarize(self, inp):
        h_sig = tf.clip_by_value((inp+1.)/2., 0, 1)
        round_out = tf.round(h_sig)
        round_fin = h_sig + tf.stop_gradient(round_out - h_sig)
        return 2.*round_fin - 1.

    #Implements y = bin(x) * w + b
    def bin_dense_layer(self, in_act, in_w, in_b, name='bin_dense_l'):
        bin_w = self.sign_binarize(in_w)
        res = tf.matmul(in_act, bin_w)
        res = tf.nn.bias_add(res, in_b) #tf.cast(in_b, dtype=tf.int32)        
        return res 
    
    #Build the TF graph for ModuloNet
    def build(self, in_act):
        epsilon = 1e-4
        in_act = tf.contrib.layers.flatten(in_act)
        #Layer 0 starts here
        #Dense layer implements y = bin(w)*x + b. Since this is the baseline implementation, we dont absorb/compose BN into the dense layer
        layer0_dense = self.bin_dense_layer(in_act, self.model['l0_w'], self.model['l0_b'], name='layer0_dense')
        #Apply BatchNorm
        layer0_dense = tf.nn.batch_normalization(layer0_dense, mean=self.model['l0_mean'], variance=self.model['l0_variance'], offset=self.model['l0_beta'], scale=self.model['l0_gamma'], variance_epsilon=1e-4, name='layer0_bn')
        #Applying baseline sign binarization.
        layer0_sig = self.sign_binarize(layer0_dense)

        #Layer 1 starts here
        layer1_dense = self.bin_dense_layer(layer0_sig, self.model['l1_w'], self.model['l1_b'], name='layer1_dense')
        layer1_dense = tf.nn.batch_normalization(layer1_dense, mean=self.model['l1_mean'], variance=self.model['l1_variance'], offset=self.model['l1_beta'], scale=self.model['l1_gamma'], variance_epsilon=1e-4, name='layer1_bn')
        layer1_sig = self.sign_binarize(layer1_dense)

        #Layer 2 starts here
        layer2_dense = self.bin_dense_layer(layer1_sig, self.model['l2_w'], self.model['l2_b'], name='layer2_dense')
        layer2_dense = tf.nn.batch_normalization(layer2_dense, mean=self.model['l2_mean'], variance=self.model['l2_variance'], offset=self.model['l2_beta'], scale=self.model['l2_gamma'], variance_epsilon=1e-4, name='layer2_bn')
        layer2_sig = self.sign_binarize(layer2_dense)

        #Layer 3 starts here
        layer3_dense = self.bin_dense_layer(layer2_sig, self.model['l3_w'], self.model['l3_b'], name='layer3_dense')
        layer3_dense = tf.nn.batch_normalization(layer3_dense, mean=self.model['l3_mean'], variance=self.model['l3_variance'], offset=self.model['l3_beta'], scale=self.model['l3_gamma'], variance_epsilon=1e-4, name='layer3_bn')
        layer3_sig = self.sign_binarize(layer3_dense)
        return layer3_sig
