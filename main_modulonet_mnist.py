
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import numpy as np
import tensorflow as tf
import bin_modulonet_mlp
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def demod(logits, k):
    dist = tf.where(logits > tf.cast(k/2, dtype=tf.int16), logits - k, logits)
    return dist

def main():
    model_path = 'autosave-modulonet-precomp-10ep.npz'#'./bnn_mnist_10ep_baseline.npz'
    arch = bin_modulonet_mlp.BinModMLP(model_path)
    #Inputs are read ranging from 0 to 1
    test_data = input_data.read_data_sets("MNIST_data/", one_hot=True).test
    #Scaling inputs to be between -127 to +127 and integers
    for i in range(test_data.images.shape[0]):
        test_data.images[i] = np.cast[np.int16](np.round(127.*(test_data.images[i] * 2 - 1)))
    for i in range(test_data.labels.shape[0]):
        test_data.labels[i] = np.cast[np.float32](np.round(127.*(test_data.labels[i] * 2 - 1)))
        
    y = tf.placeholder(tf.float32, [None, 10])
    inp_placeholder = tf.placeholder(tf.int16, [None, 28*28])
    res = arch.build(inp_placeholder)
    res1 = demod(res, 4096)
    res1 = tf.cast(res1, tf.float32)
    #loss = tf.reduce_mean(tf.square(tf.maximum(0., 1.-(y*res1))))
    correct_pred = tf.equal(tf.argmax(res1, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #new_w, new_b = arch._precompute()
    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            #new_params = sess.run([new_w, new_b])
            #np.savez('autosave-modulonet-precomp-10ep.npz', l0_w=new_params[0][0], l0_bnew=new_params[1][0], l1_w=new_params[0][1], l1_bnew=new_params[1][1], l2_w=new_params[0][2], l2_bnew=new_params[1][2], l3_w=new_params[0][3], l3_bnew=new_params[1][3])
            run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            hist = sess.run([accuracy], feed_dict={inp_placeholder: test_data.images, y: test_data.labels}, options=run_opt, run_metadata = run_metadata)
            print("-------------------------------------")
            print("Test accuracy: %f" % hist[0]) #Loss: %f" % (hist[0], hist[1]))
            print("-------------------------------------")
            #export_graph = tf.summary.FileWriter('./logs/bnn_inf_graph/', graph=sess.graph)
            #export_graph.add_run_metadata(run_metadata, 'run0')
            #export_graph.close()

if __name__=='__main__':
    main()
