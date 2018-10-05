####################################################
# Text classification with CNN - train (character level)
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.07.
####################################################



#######################################################################
### For escape tensorflow early stop error ( CTRL + C )
#######################################################################
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
#####################################
# Import modules
#####################################
from Ch01_Data_load import data_load, utils
from Ch01_Data_load import Jaso_mapping_utils as jmu
from Ch02_TextCNN.Text_CNN_config import *
from Ch02_TextCNN.Text_CNN_model import *
information = ''
FLAGS.WRITER += information



################################################################################
# DATA LOAD
################################################################################
#TRAIN_DOC, TRAIN_LABEL, TRAIN_LABEL_POS, TRAIN_LABEL_NEG, TEST_DOC, TEST_LABEL, TEST_LABEL_POS, TEST_LABEL_NEG = data_load.data_load()
TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL, LABEL_IDX = data_load.digi_data_load()
class_num = TRAIN_LABEL.shape[1]
FLAGS.NUM_OF_CLASS = class_num
JM = utils.lookup_JM(FLAGS.INPUT_WIDTH, FLAGS.INPUT_DEPTH)


################################################################################
# Start Session / Network Scratch / Save Check Point
################################################################################
# Start Session
sess = tf.Session()
print("Session Ready!")
model = MODEL(sess=sess, JM=JM, FLAGS=FLAGS)

# Initialization
sess.run(tf.global_variables_initializer())
model.JM.init_table(sess)



################################################################################
# Let's Train!!
################################################################################
# 한 epoch 당 iteration 횟수
Num_of_Iterlation = np.shape(TRAIN_DOC)[0] // FLAGS.BATCH_SIZE
epoch = 0
for i in range(Num_of_Iterlation * FLAGS.NUM_OF_EPOCH):
    # i=0
    if i % Num_of_Iterlation == 0:
        epoch += 1
        FLAGS.lr_value *= FLAGS.lr_decay

    ################################################################
    # Training batch OPTIMIZE
    ################################################################
    #index = utils.sampler(LABEL_POS=TRAIN_LABEL_POS, LABEL_NEG=TRAIN_LABEL_NEG, BATCH_SIZE=FLAGS.BATCH_SIZE)
    index = np.random.choice(range(0, len(TRAIN_DOC)), FLAGS.BATCH_SIZE)
    batch_input, batch_label = utils.generate_batch_jaso(INDEX=index, MODEL=model, DOC=TRAIN_DOC,
                                                         LABEL=TRAIN_LABEL, MAXLEN=FLAGS.INPUT_WIDTH, SESS=sess)

    _ = sess.run([model.optm],
                 feed_dict={model.X: batch_input,
                            model.Y: batch_label,
                            model.LEARNING_RATE: FLAGS.lr_value,
                            model.TRAIN_PH: True})


    ################################################################
    # Calculate Train & Test Loss, Accuracy & Summary / Print
    ################################################################
    if i % 10 == 0:
        ################################################################
        # Train batch LOSS CHECK
        ################################################################
        #index = utils.sampler(LABEL_POS=TRAIN_LABEL_POS, LABEL_NEG=TRAIN_LABEL_NEG, BATCH_SIZE=FLAGS.BATCH_SIZE)
        index = np.random.choice(range(0, len(TRAIN_DOC)), FLAGS.BATCH_SIZE)
        batch_input, batch_label = utils.generate_batch_jaso(INDEX=index, MODEL=model, DOC=TRAIN_DOC,
                                                             LABEL=TRAIN_LABEL, MAXLEN=FLAGS.INPUT_WIDTH, SESS=sess)
        proba, tr_loss, tr_acc, tr_merged = sess.run([model.y_proba, model.cross_entropy, model.accuracy, model.merge],
                                             feed_dict={model.X: batch_input,
                                                        model.Y: batch_label,
                                                        model.LEARNING_RATE: FLAGS.lr_value,
                                                        model.TRAIN_PH: False})

        pred_idx = np.apply_along_axis(np.argmax, 1, proba)
        real_idx = np.apply_along_axis(np.argmax, 1, batch_label)
        print('[ TRAINING ]')
        try:
            pos_idx = np.where(np.equal(pred_idx, real_idx) == True)[0][0]
            neg_idx = np.where(np.equal(pred_idx, real_idx) == False)[0][0]


            print('Positive Case:\t', TRAIN_DOC[index[pos_idx]], '\t->\t', LABEL_IDX[np.argmax(proba[pos_idx])],
                  '({0:.2f})\t'.format(round(max(proba[pos_idx]), 3)), LABEL_IDX[np.argmax(batch_label[pos_idx])])

            print('Negative Case:\t', TRAIN_DOC[index[neg_idx]], '\t->\t', LABEL_IDX[np.argmax(proba[neg_idx])],
                  '({0:.2f})\t'.format(round(max(proba[neg_idx]), 3)), LABEL_IDX[np.argmax(batch_label[neg_idx])],
                  '({0:.2f})\t'.format(proba[neg_idx][np.argmax(batch_label[neg_idx])])
                  )
            print()
        except:
            print('No Positive Case or No Negative Case')



        ################################################################
        # Test batch LOSS CHECK
        ################################################################
        index = np.random.choice(range(0, len(TEST_DOC)), 128)
        batch_input, batch_label = utils.generate_batch_jaso(INDEX=index, MODEL=model, DOC=TEST_DOC,
                                                             LABEL=TEST_LABEL, MAXLEN=FLAGS.INPUT_WIDTH, SESS=sess)
        proba, ts_loss, ts_acc, ts_merged = sess.run([model.y_proba, model.cross_entropy, model.accuracy, model.merge],
                                              feed_dict={model.X: batch_input,
                                                         model.Y: batch_label,
                                                         model.LEARNING_RATE: FLAGS.lr_value,
                                                         model.TRAIN_PH: False})

        pred_idx = np.apply_along_axis(np.argmax, 1, proba)
        real_idx = np.apply_along_axis(np.argmax, 1, batch_label)


        pos_idxs = np.where(np.equal(pred_idx, real_idx) == True)[0]
        neg_idxs = np.where(np.equal(pred_idx, real_idx) == False)[0]
        print('[ TEST ]')
        try:
            pos_idx = pos_idxs[0]
            neg_idx = neg_idxs[0]

            # desc = """
            #             size:{}, correct:{}, wrong:{}, acc:{}, ts_loss:{}, ts_acc:{}
            #         """.format(index.shape[0], pos_idxs.shape[0], neg_idxs.shape[0],
            #                    round(pos_idxs.shape[0] / index.shape[0] * 100, 3),
            #                    ts_loss, ts_acc)
            # print(desc)

            print('Positive Case:\t', TEST_DOC[index[pos_idx]], '\t->\t', LABEL_IDX[np.argmax(proba[pos_idx])],
                  '({0:.2f})\t'.format(round(max(proba[pos_idx]), 3)), LABEL_IDX[np.argmax(batch_label[pos_idx])])

            print('Negative Case:\t', TEST_DOC[index[neg_idx]], '\t->\t', LABEL_IDX[np.argmax(proba[neg_idx])],
                  '({0:.2f})\t'.format(round(max(proba[neg_idx]), 3)), LABEL_IDX[np.argmax(batch_label[neg_idx])],
                  '({0:.2f})\t'.format(proba[neg_idx][np.argmax(batch_label[neg_idx])])
                  )
            print()
        except:
            print('No Positive Case or No Negative Case')

        model.train_writer.add_summary(tr_merged, i)
        model.test_writer.add_summary(ts_merged, i)

        ################################################################
        # Print
        ################################################################

        print("Iter: {iter:08} / Epoch: {EP} |##| LR: {LR:0.15f} |##|  tr_LOSS: {tr_LOSS:0.8f} |##|  tr_acc: {tr_ACC:0.8f} |##|  ts_LOSS: {ts_loss:0.8f} |##|  ts_acc: {ts_acc:0.8f}".format(
                iter=i, EP=epoch, LR=FLAGS.lr_value, tr_LOSS=tr_loss, tr_ACC=tr_acc, ts_loss=ts_loss, ts_acc=ts_acc))
        FLAGS.Check_Loss = FLAGS.Check_Loss[1:]
        FLAGS.Check_Loss.append(round(tr_loss, 4))
        #print(FLAGS.Check_Loss)



################################################################################
# Save parameters
################################################################################
# Save Weights
if "Saver" not in os.listdir("./"):
    os.makedirs("./Saver")
if FLAGS.WRITER not in os.listdir("./Saver"):
    os.makedirs("./Saver/{}".format(FLAGS.WRITER))
saver = tf.train.Saver()
saver.save(sess, "./Saver/{}/{}.ckpt".format(FLAGS.WRITER, FLAGS.WRITER))

# tensorboard
# cmd창에서 summary 있는 폴더로 가서
# tensorboard --logdir=./
# tensorboard --logdir=./ --host=localhost --port=8088
