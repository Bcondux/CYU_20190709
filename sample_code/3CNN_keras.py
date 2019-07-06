# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# Create the neural network
def conv_net(x, n_classes, reuse=False):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.keras.layers.Conv2D(32, 5, activation=tf.nn.relu)(x)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.keras.layers.MaxPool2D(2, 2)(conv1)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu)(conv1)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.keras.layers.MaxPool2D(2, 2)(conv2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.keras.layers.Flatten()(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.keras.layers.Dense(1024)(fc1)

        # Output layer, class prediction
        out = tf.keras.layers.Dense(n_classes)(fc1)

    return out

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

# Construct model
logits = conv_net(X, num_classes)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256]}))