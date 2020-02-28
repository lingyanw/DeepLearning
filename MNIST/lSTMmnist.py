
# coding: utf-8

# In[22]:


import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np 
#from _future_ import print_function

from tensorflow.examples.tutorials.mnist import input_data
tf.reset_default_graph()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
result_dir = './results2/'
learningRate = 0.001
trainingIters = 10000
batchSize = 128
displayStep = 200

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 512 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(x, nSteps, 0) #configuring so you can get it as needed for the 28 pixels

    #lstmCell = rnn_cell.BasicRNNCell(nHidden) #find which lstm to use in the documentation
    #lstmCell = rnn_cell.BasicLSTMCell(nHidden,forget_bias = 1.0)
    lstmCell = rnn_cell.GRUCell(nHidden)

    outputs, states = rnn.static_rnn(lstmCell, x, dtype=tf.float32) #for the rnn where to get the output and hidden state 

    return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)
prediction = tf.nn.softmax(pred)
#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# summary to visualize filters 
# Add a scalar summary for the snapshot loss.
f1 = tf.summary.scalar(cost.op.name, cost)
f2 = tf.summary.scalar(accuracy.op.name, accuracy)
# Build the summary operation based on the TF collection of Summaries.
#f3 = tf.summary.image('L1_filter', put_kernels_on_grid (W_conv1),max_outputs=1)
summary_op = tf.summary.merge_all()
# Add the variable initializer Op.
init = tf.initialize_all_variables()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.summary.FileWriter(result_dir, sess.graph)




init = tf.initialize_all_variables()

#with sess as sess:
sess.run(init)
step = 1

while step < trainingIters:
    batchX, batchY = mnist.train.next_batch(batchSize)
    batchX = batchX.reshape((batchSize, nSteps, nInput))

    sess.run(optimizer, feed_dict={x: batchX, y: batchY})
        #print(step)

    if step % displayStep == 0:
        print(step)
        loss, acc = sess.run([cost,accuracy], feed_dict={x: batchX, y: batchY})
        print("Iter " + str(step) + ", Minibatch Loss= " +                 "{:.6f}".format(loss) + ", Training Accuracy= " +                 "{:.5f}".format(acc))
        summary_str1,summary_str2 = sess.run([f1,f2], feed_dict={x: batchX, y: batchY})
        summary_writer.add_summary(summary_str1, step)
        summary_writer.add_summary(summary_str2, step)
        summary_writer.flush()
#     if step % 5000 == 0 or step == trainingIters:
#         checkpoint_file = os.path.join(result_dir, 'checkpoint')
#         saver.save(sess, checkpoint_file, global_step=i)
    step +=1
print('Optimization finished')

testData = mnist.test.images.reshape((-1, nSteps, nInput))
testLabel = mnist.test.labels
print("Testing Accuracy:",     sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
    
    
    
    
    


# In[19]:


trainingIters


# In[20]:


displayStep

