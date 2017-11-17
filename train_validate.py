from tensorflow.contrib.layers import flatten
from parameters import *
from helper_functions import norm
from lenet import LeNet
from sklearn.utils import shuffle

# Define the graph for the network
forward_prop = LeNet(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = forward_prop, labels = one_hot_y))#CHANGED
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(forward_prop, 1), tf.argmax(one_hot_y, 1))#CHANGED
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def eval_data(dataset, labels):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    # If dataset.num_examples is not divisible by BATCH_SIZE
    # the remainder will be discarded.
    # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
    # steps_per_epoch = 55000 // 64 = 859
    # num_examples = 859 * 64 = 54976
    #
    # So in that case we go over 54976 examples instead of 55000.
    steps_per_epoch = dataset.shape[0] // BATCH_SIZE #CHANGED
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    sess = tf.get_default_session()
    for step in range(steps_per_epoch):
        batch_start = step * BATCH_SIZE
        batch_x, batch_y = X_train[batch_start: batch_start + BATCH_SIZE], y_train[batch_start: batch_start + BATCH_SIZE]
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_acc/num_examples


if __name__ == '__main__':
    # Load data

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps_per_epoch = X_train.shape[0] // BATCH_SIZE #CHANGED
        num_examples = steps_per_epoch * BATCH_SIZE
        
        # Train model
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train, random_state=0)
            X_valid, y_valid = shuffle(X_valid, y_valid, random_state=0)
            for step in range(steps_per_epoch):
                # Input data is already normalized - range(0,1)
                batch_start = step * BATCH_SIZE

                batch_x, batch_y = X_train[batch_start: batch_start + BATCH_SIZE], y_train[batch_start: batch_start + BATCH_SIZE]

                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            val_loss, val_acc = eval_data(X_valid, y_valid) #CHANGED
            print("EPOCH {} ...".format(i+1))
            print("Validation loss = {:.3f}".format(val_loss))
            print("Validation accuracy = {:.3f}".format(val_acc))
            print()

        # Evaluate on the test data
        if final:
            test_loss, test_acc = eval_data(X_test, y_test) #CHANGED
            print("Test loss = {:.3f}".format(test_loss))
            print("Test accuracy = {:.3f}".format(test_acc))

