from tensorflow.contrib.layers import flatten
from plot import plot_validation
from parameters import *
from lenet import LeNet
from sklearn.utils import shuffle
from datetime import datetime
import os
import copy



def savedata(train_loss, train_acc, validation_acc):

    ''' Method to save data from the training session '''

    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    directory = 'tf_data'

    str_lr = str(lr_3) 
    str_b = str(beta)

    logdir = '{}/run-{}_b{}_k{}_lr{}_{}_{}_{}_{}_{}_rotate/'.format(directory,now, str_b[2:], keep_train*10, str_lr[2:], lr_thres_1*100, lr_thres_2*100, layer_size['fc_1'], layer_size['fc_2'], layer_size['fc_3'])

    # Summary scalars for tensorboard
    with tf.name_scope("train_assessment"):
        tf.summary.scalar('Train Loss', train_loss)
        tf.summary.scalar('Train Accuracy', train_acc)
    with tf.name_scope("valid_assessment"):
        tf.summary.scalar('Validation Accuracy', validation_acc)

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    return logdir, file_writer

with tf.name_scope("batch_training"):
    # Define the graph for the network
    with tf.name_scope("forward_propogation"):
        forward_prop, c1, c2, c3, f1, f2 = LeNet(x, keep_prob)
    with tf.name_scope("calc_cost"):
        ce = tf.nn.softmax_cross_entropy_with_logits(logits = forward_prop, labels = one_hot_y)
        soft = tf.nn.softmax(logits = forward_prop)
        loss_op = tf.reduce_mean(ce)#CHANGED
    
    with tf.name_scope("optimizer"):
        opt = tf.train.AdamOptimizer(learning_rate = learning_rate)

    with tf.name_scope("calc_accuracy"):
        # If penalizing large network weights, run the following code
        if l2_regularize:
            with tf.name_scope("regularization"):
                reg = tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(weights['wc3']) + tf.nn.l2_loss(weights['wf1']) + \
                tf.nn.l2_loss(weights['wf2']) + tf.nn.l2_loss(weights['wf3']) + tf.nn.l2_loss(weights['out'])
            with tf.name_scope("calc_loss"):
                loss_op = tf.reduce_mean(loss_op + beta * reg)

        with tf.name_scope("train"):
            train_op = opt.minimize(loss_op)
        with tf.name_scope("accuracy"):
            prediction = tf.argmax(forward_prop, 1)
            label = tf.argmax(one_hot_y, 1)
            correct_prediction = tf.equal(prediction, label)
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.name_scope("validation"):
        forward_val = LeNet(x_valid_p, keep_prob)[0]
        correct_prediction_val = tf.equal(tf.argmax(forward_val, 1), tf.argmax(one_hot_y_val, 1))
        accuracy_op_val = tf.reduce_mean(tf.cast(correct_prediction_val, tf.float32))

if not pre_trained:
    # Creat directory and file_writer to save training summary information
    logdir, file_writer  = savedata(loss_op, accuracy_op, accuracy_op_val)

# Merge all summaries so they can be addressed as one object
merged = tf.summary.merge_all()

def eval_data(dataset, labels, plot_layers = False, pre = False, int_im = False):
    """
    Given a dataset as input returns the loss and accuracy.
    """

    steps_epoch = dataset.shape[0] // BATCH_SIZE
    num = steps_epoch * BATCH_SIZE

    if int_im:
        steps_epoch = 1


    total_acc, total_loss = 0, 0
    correct_list = []
    with tf.Session() as sess:
        if pre:
            #sess = tf.get_default_session()
            saver = tf.train.Saver()
            #saver = tf.train.import_meta_graph('tmp_rot/model.ckpt.meta')

            # We can now access the default graph where all our metadata has been loaded
            graph = tf.get_default_graph()

            saver.restore(sess, "best_model/model.ckpt")

            optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        else:
            sess = tf.get_default_session()
        for step in range(steps_epoch):
            if int_im:
                batch_start = 0
                num = 1
                batch_x, batch_y = dataset, labels
                loss, acc, correct, pred, lab, s = sess.run([loss_op, accuracy_op, correct_prediction, prediction, label, soft], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                total_acc, total_loss = acc, loss

                #print('Forward: {}'.format(fw))
                #print('prediction: {}'.format(pred))
                #print('label: {}'.format(lab))

                ## Plot the softmax outputs for the 5 example images ##

                colors = {1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm'}
                labs = {}
                flat_names = [item for sublist in signnames for item in sublist]

                fig, ax = plt.subplots()
                for i, prop in enumerate(s):
                    # Iterate through the network outputs for each image
                    labs[i + 1]= ax.plot(range(0,43), s[i], colors[i + 1])
                    plt.ylabel('Network Output')
                    plt.xlabel('Label Number')
                    plt.grid()
                locs, labels = plt.xticks()
                plt.xticks(range(0,43), flat_names, rotation='vertical')
                ledge = ['Priority Road (12)', 'Turn Left Ahead (34)', 'End of all speed (32)', 'Yield (13)', 'Right of way @ Intersection (11)']
                ax.legend(ledge)
                mng = plt.get_current_fig_manager()
                mng.full_screen_toggle()
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.5)
                

                ## For each of the 5 example images, print the top 5 predictions ##
                x_max = 0
                prob_max = 0

                # Iterate through the images
                for i, prop in enumerate(s):
                    # Iterate through the predictions
                    tmp = list(copy.deepcopy(prop))
                    tmp_2 = copy.deepcopy(flat_names)
                    print('\nPredicting: {}'.format(ledge[i]))
                    # Find the 5 highest predictions
                    for g in range(5):
                        for c, prob in enumerate(tmp):
                            if prob > prob_max:
                                x_max = c
                                prob_max = prob
                        print('{}. {}: {:.3f}'.format(g + 1, tmp_2[x_max],prob_max))
                        tmp.pop(x_max)
                        tmp_2.pop(x_max)
                        x_max, prob_max = 0, 0
                plt.show()

            else:
                batch_start = step * BATCH_SIZE
                batch_x, batch_y = dataset[batch_start: batch_start + BATCH_SIZE], labels[batch_start: batch_start + BATCH_SIZE]
                loss, acc, correct = sess.run([loss_op, accuracy_op, correct_prediction], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})                
                #print('{}, {}, {}'.format(loss, acc, correct))
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

            # Run once on the training set
            train_loss, train_acc, train_correct, sum_t = eval_data(X_train, y_train, pre = pre_trained)
            print("\nTraining loss = {:.3f}".format(train_loss))
            print('Accuracy on training set: {:.3f}\n'.format(train_acc))

            # Run once on the validation set
            val_loss, val_acc, val_correct, sum_v = eval_data(X_valid, y_valid, pre = pre_trained)
            print("\nValidation loss = {:.3f}".format(val_loss))
            print('Accuracy on validation set: {:.3f}\n'.format(val_acc))

            # Run on test set
            if final:
                 test_loss, test_acc, test_correct, sum_test = eval_data(X_test, y_test, X_test.shape[0], pre = pre_trained)
                 print("\nTest loss = {:.3f}".format(test_loss))
                 print("Test accuracy = {:.3f}\n".format(test_acc))

            # Run on images from internet
            if internet:
                # Run the 5 downloaded images on the network
                int_loss, int_acc, int_correct, sum_i = eval_data(X_int, y_int, X_int.shape[0], pre = pre_trained, int_im = True)
                print("\nLoss internet images = {:.3f}".format(int_loss))
                print("Accuracy on internet images = {:.3f}\n".format(int_acc))

        else:
            # Create saver object to save the network information to be used in another session
            saver = tf.train.Saver()
            # Train and validate the model over a specified number of epochs
            for i in range(EPOCHS):
                # Shuffle the data once per epoch
                X_train, y_train = shuffle(X_train, y_train, random_state=0)
                
                for step in range(steps_per_epoch):
                    short_val_acc = []
                    # Optimize network over each epoch
                    batch_start = step * BATCH_SIZE
                    # Pull a batch from the shuffled training set
                    batch_x, batch_y = X_train[batch_start: batch_start + BATCH_SIZE], y_train[batch_start: batch_start + BATCH_SIZE]
                    # Decrease the learning rate as the validation accuracy increases
                    if val_acc < lr_thres_1:#changed from 0.8
                        lr = lr_1
                    elif val_acc < lr_thres_2:#changed from 0.94
                        lr = lr_2
                    else:   
                        lr = lr_3#updated

                    # Minimize the loss using the chosen learning rate
                    loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_train, learning_rate : lr})

                    if step % 50:
                        X_valid, y_valid = shuffle(X_valid, y_valid, random_state=0)
                        # Summarize after every epoch
                        (summary, t_a, v_a) = sess.run([merged, accuracy_op, accuracy_op_val], feed_dict = {x: batch_x, y: batch_y, x_valid_p: X_valid[0:500], y_valid_p: y_valid[0:500], keep_prob: 1.0})
                        s_t = i * steps_per_epoch + step
                        list_train_acc.append(t_a)
                        list_val_acc.append(v_a)
                        file_writer.add_summary(summary, s_t)
                        list_lr.append(lr)
                        short_val_acc.append(v_a)

                #wrong = sum([1 for element in v_cor if element == False])
                print("\nEPOCH {} ...".format(i+1))
                #print("Validation loss = {:.3f}".format(v_l))
                m = np.mean(short_val_acc)
                print("Validation accuracy (mean over epoch) = {:.3f}".format(m))
                #print('Incorrect: {}'.format(wrong))

                # Check if all 
                #if  False not in v_cor:
                #    print('Exiting training as validation set has been guessed correctly.')
                #    print('val_correct: {}'.format(v_cor))
                #    break
                if m >= 0.99:
                    save_path = saver.save(sess, os.path.join(os.getcwd(), 'tmp_rot','model.ckpt'))
                    print("Model saved in file: %s" % save_path)
                    break
            else:
                print('99% not reached - here is the final model')
                save_path = saver.save(sess, os.path.join(os.getcwd(), 'tmp_rot','model.ckpt'))
                print("Model saved in file: %s" % save_path)

        if not pre_trained:
            valid_loss_full, valid_acc_full, correct_list, sum_correct = eval_data(X_valid, y_valid)
            print("\nValidation Full Set loss = {:.3f}".format(valid_loss_full))
            print("Validation Full Set accuracy = {:.3f}".format(valid_acc_full))
            print("Validation Full Set incorrect = {:.3f}".format(wrong))

        if not pre_trained:
            file_writer.close()