import tensorflow as tf
X = tf.Variable([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.4, 0.6, 0.8, 1.0]]) # shape(2,5)
F = tf.Variable([[2.5, 1.4, 0.2, 0.2, 4.5]]) # shape(1, 5) transpose:(5,1)
f = tf.Variable([[25, 14, 2, 2, 45]], dtype=tf.float32) # shape(1, 5) transpose:(5,1)
FM = tf.Variable([[2.5, 1.4, 0.2, 0.2, 4.5], [25, 14, 2, 2, 45]])# if we merge the F and f into one tensor (shape of (2,5))
sess = tf.Session()
sess.run(X.initializer)
sess.run(X) # show the result of X
# initialize the other variables
sess.run(F.initializer)
sess.run(f.initializer)
sess.run(FM.initializer)
sess.run(tf.matmul(X, tf.transpose(F)))
sess.run(tf.matmul(X, tf.transpose(f)))
sess.run(tf.matmul(X, tf.transpose(FM)))
sess.close()

