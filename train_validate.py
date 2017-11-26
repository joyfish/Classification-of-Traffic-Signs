from tensorflow.contrib.layers import flatten
from plot import plot_validation
from parameters import *
from lenet import LeNet
from sklearn.utils import shuffle
from datetime import datetime
import os

def savedata(loss, acc):

    ''' Method to save data from the training session '''

    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    directory = 'tf_data'
    logdir = '{}/run-{}_b{}_k{}_lr{}_{}_{}_{}_{}/'.format(directory,now, '00001', '3', '00001', '90', '94', '300', '100')

    # Summary scalars for tensorboard
    with tf.name_scope("train_assessment"):
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', acc)

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    return logdir, file_writer


with tf.name_scope("batch_training"):
    # Define the graph for the network
    with tf.name_scope("forward_propogation"):
        forward_prop, c1, c2, c3, f1, f2 = LeNet(x, keep_prob)
    with tf.name_scope("calc_cost"):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = forward_prop, labels = one_hot_y))#CHANGED
    
    with tf.name_scope("optimizer"):
        opt = tf.train.AdamOptimizer(learning_rate = learning_rate)

    with tf.name_scope("calc_accuracy"):
        # If penalizing large network weights, run the following code
        if l2_regularize:
            with tf.name_scope("regularization"):
                reg = tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(weights['wc3']) + tf.nn.l2_loss(weights['wf1']) + \
                tf.nn.l2_loss(weights['wf2']) + tf.nn.l2_loss(weights['out'])
            with tf.name_scope("calc_loss"):
                loss_op = tf.reduce_mean(loss_op + beta * reg)

        with tf.name_scope("train"):
            train_op = opt.minimize(loss_op)
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(forward_prop, 1), tf.argmax(one_hot_y, 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Creat directory and file_writer to save training summary information
logdir, file_writer  = savedata(loss_op, accuracy_op)

# Create saver object to save the network information to be used in another session
saver = tf.train.Saver()

# Merge all summaries so they can be addressed as one object
merged = tf.summary.merge_all()


def eval_data(dataset, labels, plot_layers = False):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    steps_epoch = dataset.shape[0] // BATCH_SIZE

    num = steps_epoch * BATCH_SIZE

    total_acc, total_loss = 0, 0
    correct_list = []
    sess = tf.get_default_session()
    for step in range(steps_epoch):
        batch_start = step * BATCH_SIZE
        batch_x, batch_y = dataset[batch_start: batch_start + BATCH_SIZE], labels[batch_start: batch_start + BATCH_SIZE]
        loss, acc, correct = sess.run([loss_op, accuracy_op, correct_prediction], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
        correct_list.extend(list(correct))
    return total_loss/num, total_acc/num, correct_list, sum(correct_list)


if __name__ == '__main__':

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps_per_epoch = X_train.shape[0] // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        val_acc = 0
        list_val_acc, list_train_acc, list_lr = [], [], []
    
        if pre_trained:
            # If pre trained, no need to retrain
            # Load trained model from hardrive
            saver.restore(sess, "/tmp/model.ckpt")
            print("Model restored.")
            # Run once on the training set
            train_loss, train_acc, train_correct, sum_t = eval_data(X_train, y_train)
            plot_validation(train_correct, y_train, string = 'Training')
            # Run once on the validation set
            val_loss, val_acc, val_correct, sum_v = eval_data(X_valid, y_valid)
            plot_validation(val_correct, y_valid)
        else:
            # Train and validate the model over a specified number of epochs
            for i in range(EPOCHS):
                # Shuffle the data once per epoch
                X_train, y_train = shuffle(X_train, y_train, random_state=0)
                X_valid, y_valid = shuffle(X_valid, y_valid, random_state=0)

                for step in range(steps_per_epoch):
                    # Optimize network over each epoch
                    batch_start = step * BATCH_SIZE
                    # Pull a batch from the shuffled training set
                    batch_x, batch_y = X_train[batch_start: batch_start + BATCH_SIZE], y_train[batch_start: batch_start + BATCH_SIZE]
                    # Decrease the learning rate as the validation accuracy increases
                    if val_acc < 0.9:#changed from 0.8
                        lr = 0.001
                    elif val_acc < 0.94:#changed from 0.94
                        lr = 0.0001
                    else:   
                        lr = 0.00001#updated

                    # Minimize the loss using the chosen learning rate
                    loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_train, learning_rate : lr})

                    if step % 50:
                        # Every 50 steps of training data, summarize thee network summary variables
                        (summary, t_a) = sess.run([merged, accuracy_op], feed_dict = {x: batch_x, y: batch_y, keep_prob: 1.0})
                        s_t = i * steps_per_epoch + step
                        list_train_acc.append(t_a)
                        file_writer.add_summary(summary, s_t)

                # After each epoch, apply this iteration of the network to the validation set
                val_loss, val_acc, val_correct, sum_v = eval_data(X_valid, y_valid)
                list_val_acc.append(val_acc)
                list_lr.append(lr)
                wrong = sum([1 for element in val_correct if element == False])
                print("\nEPOCH {} ...".format(i+1))
                print("Validation loss = {:.3f}".format(val_loss))
                print("Validation accuracy = {:.3f}".format(val_acc))
                print('Incorrect: {}'.format(wrong))

                # Check if all 
                if  False not in val_correct:
                    print('Exiting training as validation set has been guessed correctly.')
                    print('val_correct: {}'.format(val_correct))
                    break
            else:    
                save_path = saver.save(sess, os.path.join(os.getcwd(), 'tmp','model.ckpt'))
                print("Model saved in file: %s" % save_path)

        x_lr = np.linspace(0, 1, len(list_lr))
        x_val = np.linspace(0, 1, len(list_val_acc)) 
        x_train = np.linspace(0, 1, len(list_train_acc)) 

        list_lr = [i * 1000 for i in list_lr]

        learn = plt.plot(x_lr, list_lr, 'r')
        val = plt.plot(x_val, list_val_acc, 'g')
        train = plt.plot(x_train, list_train_acc, 'b')
        plt.legend([learn, val, train], ['Learning Rate * 1000', 'Val Accuracy', 'Train Accuracy'])
        plt.show()
        plt.savefig('foo.png')


        # Evaluate on the test data if model has been finalized
        if internet:
            # Run the 5 downloaded images on the network
            int_loss, int_acc, int_correct, sum_i = eval_data(X_int, y_int, plot_layers = True)
            print("Loss internet images = {:.3f}".format(int_loss))
            print("Accuracy on internet images = {:.3f}".format(int_acc))

        # Evaluate on the test data if model has been finalized
        if final:
            test_loss, test_acc, test_correct, sum_t = eval_data(X_test, y_test)
            print("Test loss = {:.3f}".format(test_loss))
            print("Test accuracy = {:.3f}".format(test_acc))

        file_writer.close()