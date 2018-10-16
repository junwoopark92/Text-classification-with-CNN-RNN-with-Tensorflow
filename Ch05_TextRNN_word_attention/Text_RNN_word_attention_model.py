####################################################
# Text classification with attention RNN - model (word level)
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.19.
####################################################



#####################################
# Import modules
#####################################
import gc
gc.collect()
from Ch05_TextRNN_word_attention.Text_RNN_word_attention_config import *
import Ch01_Data_load.utils as utils



################################################################################
# Network Scratch!
################################################################################
class MODEL():

    def __init__(self, sess, FLAGS):

        ####################################################
        # Jaso mapping
        ####################################################
        self.sess = sess

        ####################################################
        # Set placeholders
        ####################################################
        self.set_placeholders()

        ####################################################
        # Network structure
        ####################################################
        self.set_network()

        ####################################################
        # Optimizer / accuracy / loss / tensorboard
        ####################################################
        self.set_ops(FLAGS.WRITER, FLAGS.WRITER_generate)


    ########################################################
    # Set placeholder
    ########################################################
    def set_placeholders(self):
        self.X_idx = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.MAXLEN])
        self.Y = tf.placeholder(tf.float32, [None, FLAGS.NUM_OF_CLASS])
        self.SEQ = tf.placeholder(tf.int32, [None])
        self.LEARNING_RATE = tf.placeholder(tf.float32)
        self.Dropout_Rate1 = tf.placeholder(tf.float32)
        self.Dropout_Rate2 = tf.placeholder(tf.float32)
        self.TRAIN_PH = tf.placeholder(tf.bool)


    ########################################################
    # Network structure
    ########################################################
    def set_network(self):

        ##########################################
        # Embedding
        ##########################################
        with tf.variable_scope('Embedding'):
            self.W = tf.get_variable(name='embedding_matrix',
                                     shape=[FLAGS.VOCAB_SIZE, FLAGS.EMBEDDING_SIZE],
                                     initializer=tf.random_normal_initializer(stddev=0.1))
            self.X = tf.nn.embedding_lookup(self.W, self.X_idx)


        ##########################################
        # Recurrent layer
        ##########################################
        with tf.variable_scope('{}_cell'.format(FLAGS.RNN_CELL)):
            multi_cells = utils.RNN_structure(FLAGS.RNN_CELL, FLAGS.RNN_HIDDEN_DIMENSION, self.Dropout_Rate1, self.Dropout_Rate2, FLAGS.N_LAYERS)
            # RNN 신경망을 생성 (SEQ를 통해 매 sentence의 길이까지만 계산을 해 효율성 증대)
            outputs, _states = tf.nn.dynamic_rnn(cell=multi_cells, inputs=self.X, dtype=tf.float32) #sequence_length=self.SEQ,

        # Attention
        with tf.variable_scope('Attention'):
            self.att_matrix, self.att_alpha = utils.attention(INPUTS=outputs, ATTENTION_SIZE=FLAGS.ATTENTION_SIZE, SEQ=self.SEQ, time_major=False, return_alphas=True)


        ##########################################
        # Fully connected network
        ##########################################
        with tf.variable_scope('FC-layer'):
            FC1 = tf.contrib.layers.fully_connected(self.att_matrix, FLAGS.FC_HIDDEN_DIMENSION, weights_initializer=utils.he_init, activation_fn=None)
            FC_act1 = tf.nn.relu(tf.layers.batch_normalization(FC1, momentum=0.9, training=self.TRAIN_PH))
            FC2 = tf.contrib.layers.fully_connected(FC_act1, FLAGS.FC_HIDDEN_DIMENSION, weights_initializer=utils.he_init, activation_fn=None)
            FC_act2 = tf.nn.relu(tf.layers.batch_normalization(FC2, momentum=0.9, training=self.TRAIN_PH))

            self.y_logits = tf.contrib.layers.fully_connected(FC_act2, FLAGS.NUM_OF_CLASS, activation_fn=None)


    ########################################################
    # Optimizer / accuracy / loss / tensorboard
    ########################################################
    def set_ops(self, WRITER, WRITER_generate):
        # loss
        with tf.variable_scope('cross_entropy'):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits, labels=self.Y, name='loss'))
            tf.summary.scalar('cross_entropy', self.cross_entropy)

        # optimizer
        with tf.variable_scope('Optimizer'):
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.optm = tf.train.RMSPropOptimizer(self.LEARNING_RATE).minimize(loss=self.cross_entropy)

        # accuracy
        with tf.variable_scope('accuracy'):
            with tf.variable_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.y_logits, 1), tf.argmax(self.Y, 1))
            with tf.variable_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc')
                tf.summary.scalar('accuracy', self.accuracy)
        print("Optimizer Ready! & Let's Train!")


        self.merge = tf.summary.merge_all()
        if WRITER_generate:
            self.train_writer = tf.summary.FileWriter("./output/{}/train_{}".format(WRITER, WRITER), self.sess.graph)
            self.test_writer = tf.summary.FileWriter("./output/{}/test_{}".format(WRITER, WRITER), self.sess.graph)
        print("Graph Ready!")

