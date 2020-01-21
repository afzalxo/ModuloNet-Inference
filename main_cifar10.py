
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import numpy as np
import tensorflow as tf
import bin_convnet
from tensorflow.keras.datasets import cifar10 as input_data
import os
import matplotlib.pyplot as plt
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def demod(logits, k):
    dist = tf.where(logits > k/2., logits - k, logits)
    return dist

def main():
    model_path = 'autosave-cifar10-baseline-20ep-8013.npz'#'autosave-cifar10-baseline-20ep-8013.npz'#'./autosave-cifar10-20ep-bn_before_maxpool-7783.npz'
    arch = bin_convnet.BinCNN(model_path)
    (_, _), (test_data, test_labels) = input_data.load_data()
    test_data = np.reshape(test_data/255. * 2 - 1, (-1, 32, 32, 3))
    test_labels = np.float32(np.eye(10)[np.hstack(test_labels)]) * 2 - 1
    test_data = test_data[0:5000,:,:,:]
    test_labels = test_labels[0:5000,:]
    y = tf.placeholder(tf.float32, [None]+list(test_labels.shape[1:]))
    inp_placeholder = tf.placeholder(tf.float32, [None]+list(test_data.shape[1:]))

    res1 = arch.build_convnet(inp_placeholder)

    loss = tf.reduce_mean(tf.square(tf.maximum(0., 1.-y*res1)))
    correct_pred = tf.equal(tf.argmax(res1, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            #run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()
            #ret = sess.run(res, feed_dict={inp_placeholder:inp}, options=run_opt, run_metadata=run_metadata)
            #print(ret)
            hist = sess.run([accuracy, loss], feed_dict={inp_placeholder: test_data, y: test_labels})#, options=run_opt, run_metadata = run_metadata)
            print("-------------------------------------")
            print("Test accuracy: %f, Loss: %f" % (hist[0], hist[1]))
            print("-------------------------------------")
            '''
            net_out = hist[15]
            plt.plot(net_out.flatten())
            plt.xlabel('Array Index')
            plt.ylabel('Bias Value')
            plt.title('Biases of Layer 2')
            plt.savefig('bias2.png')
            plt.clf()
            '''
            #export_graph = tf.summary.FileWriter('./logs/bnn_inf_graph/', graph=sess.graph)
            #export_graph.add_run_metadata(run_metadata, 'run0')
            #export_graph.close()

if __name__=='__main__':
    main()
