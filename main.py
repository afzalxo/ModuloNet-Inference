
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import numpy as np
import tensorflow as tf
import binnn
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def demod(logits, k):
    dist = tf.where(logits > k/2., logits - k, logits)
    return dist


def main():
    model_path = './bnn_mnist_10ep_baseline.npz'# autosave-reduced-bn-10ep.npz'
    arch = binnn.BinNN(model_path)
    test_data = input_data.read_data_sets("MNIST_data/", one_hot=True).test
    inp = test_data.images[0:2] * 2 - 1
    for i in range(test_data.images.shape[0]):
        test_data.images[i] = np.round(127.*(test_data.images[i] * 2 - 1))
    for i in range(test_data.labels.shape[0]):
        test_data.labels[i] = np.round(127.*(test_data.labels[i] * 2 - 1))
        
    y = tf.placeholder(tf.float32, [None, 10])
    inp_placeholder = tf.placeholder(tf.float32, [None, 28*28])
    res, bn, l2_mod, l2_bn , psi0, phi0, psi1, phi1, psi2, phi2, biases0, biases1, biases2 = arch.build(inp_placeholder)
    res1 = res#demod(res, 8.)
    loss = tf.reduce_mean(tf.square(tf.maximum(0., 1.-y*res1)))
    #lo1 = tf.where(res > 4., res, -999.0)
    #lo2 = tf.where(res <= 4., res, -999.0)
    correct_pred = tf.equal(tf.argmax(res1, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            #ret = sess.run(res, feed_dict={inp_placeholder:inp}, options=run_opt, run_metadata=run_metadata)
            #print(ret)
            hist = sess.run([accuracy, loss, res, bn, res1, l2_mod, l2_bn, psi0, phi0, psi1, phi1, psi2, phi2, biases0, biases1, biases2], feed_dict={inp_placeholder: test_data.images, y: test_data.labels}, options=run_opt, run_metadata = run_metadata)
            print("-------------------------------------")
            print("Test accuracy: %f, Loss: %f" % (hist[0], hist[1]))
            bn_out = hist[3]
            plt.hist(bn_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('Output of bn')
            plt.savefig('bn_out.png')
            plt.clf()
            net_out = hist[2]
            plt.hist(net_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('output of the network: l3 mod k')
            plt.savefig('res.png')
            plt.clf()
            net_out = hist[4]
            plt.hist(net_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('res1')
            plt.savefig('res1.png')
            plt.clf()
            net_out = hist[5]
            plt.hist(net_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('Output of modulo k layer 2')
            plt.savefig('l2_mod4096.png')
            plt.clf()
            net_out = hist[6]
            plt.hist(net_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('Output of BN2 layer2')
            plt.savefig('l2_bn2.png')
            plt.clf()
            net_out = hist[7]
            plt.hist(net_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('psi0')
            plt.savefig('psi0.png')
            plt.clf()
            net_out = hist[8]
            plt.hist(net_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('phi0')
            plt.savefig('phi0.png')
            plt.clf()
            net_out = hist[9]
            plt.hist(net_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('psi1')
            plt.savefig('psi1.png')
            plt.clf()
            net_out = hist[10]
            plt.hist(net_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('phi1')
            plt.savefig('phi1.png')
            plt.clf()
            net_out = hist[11]
            plt.hist(net_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('psi2')
            plt.savefig('psi2.png')
            plt.clf()
            net_out = hist[12]
            plt.hist(net_out.flatten(), bins=64)
            plt.xlabel('Value Bin')
            plt.ylabel('Number of Occurances')
            plt.title('phi2')
            plt.savefig('phi2.png')
            plt.clf()
            net_out = hist[13]
            plt.plot(net_out.flatten())
            plt.xlabel('Array Index')
            plt.ylabel('Bias Value')
            plt.title('Biases of Layer 0')
            plt.savefig('bias0.png')
            plt.clf()
            net_out = hist[14]
            plt.plot(net_out.flatten())
            plt.xlabel('Array Index')
            plt.ylabel('Bias Value')
            plt.title('Biases of Layer 1')
            plt.savefig('bias1.png')
            plt.clf()
            net_out = hist[15]
            plt.plot(net_out.flatten())
            plt.xlabel('Array Index')
            plt.ylabel('Bias Value')
            plt.title('Biases of Layer 2')
            plt.savefig('bias2.png')
            plt.clf()
            export_graph = tf.summary.FileWriter('./logs/bnn_inf_graph/', graph=sess.graph)
            export_graph.add_run_metadata(run_metadata, 'run0')
            export_graph.close()

if __name__=='__main__':
    main()
