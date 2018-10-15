import os
import argparse
import json
import pandas as pd
from sklearn.metrics import f1_score
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from Ch01_Data_load import data_load, utils
from Ch01_Data_load import Jaso_mapping_utils as jmu
from Ch02_TextCNN.Text_CNN_config import *
from Ch02_TextCNN.Text_CNN_model import *


def json2intent(text):
    j = json.loads(text)
    domains = j['domain_info']
    intent = []
    for domain in domains:
        try:
            speech_act = domain['speech_act']
            if not speech_act == 'not_understanding':
                intent.append(domain['domain_name'] + '_P' + speech_act)
        except Exception as e:
            # print(e)
            continue

    if len(intent) > 0:
        return intent[0]
    else:
        return None


def parse_args():
    desc = "Character level TextCNN"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input', type=str, default='tmp_not_under.txt', help='[a.txt, ,tmp.txt, tmp_not_under.txt]')
    parser.add_argument('--inputdir', type=str, default='./testdata/')
    parser.add_argument('--printtest', type=str, default='True')

    return parser.parse_args()


def main():
    args = parse_args()
    if args is None:
        exit()

    # test file
    filename = args.inputdir + args.input


    # init model config
    #TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL, LABEL_IDX = data_load.digi_data_load()
    TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL, LABEL_IDX = data_load.testcase_shuffle_data_load()
        #data_load.testcase_add_data_load()
    class_num = TRAIN_LABEL.shape[1]
    FLAGS.NUM_OF_CLASS = class_num
    JM = utils.lookup_JM(FLAGS.INPUT_WIDTH, FLAGS.INPUT_DEPTH)

    # Start Session
    sess = tf.Session()
    print("Session Ready!")
    model = MODEL(sess=sess, JM=JM, FLAGS=FLAGS)

    # Initialization
    sess.run(tf.global_variables_initializer())
    model.JM.init_table(sess)

    # Restore parameter
    saver = tf.train.Saver()
    saver.restore(sess, "./Saver/{}/{}.ckpt".format(FLAGS.WRITER, FLAGS.WRITER))

    if args.printtest == 'True':
        index = np.array(range(0, len(TEST_DOC)))
        batch_input, batch_label = utils.generate_batch_jaso(INDEX=index, MODEL=model, DOC=TEST_DOC,
                                                             LABEL=TEST_LABEL, MAXLEN=FLAGS.INPUT_WIDTH, SESS=sess)

        proba, ts_loss, ts_acc, ts_merged = sess.run([model.y_proba, model.cross_entropy, model.accuracy, model.merge],
                                                     feed_dict={model.X: batch_input,
                                                                model.Y: batch_label,
                                                                model.LEARNING_RATE: FLAGS.lr_value,
                                                                model.TRAIN_PH: False})

        pred_idx = np.apply_along_axis(np.argmax, 1, proba)
        real_idx = np.apply_along_axis(np.argmax, 1, batch_label)

        pos_idx = np.where(np.equal(pred_idx, real_idx) == True)[0]
        neg_idx = np.where(np.equal(pred_idx, real_idx) == False)[0]

        print('[ TEST ]')
        desc = """
            size:{}, correct:{}, wrong:{}, acc:{}, f1_score:{}, ts_loss:{}, ts_acc:{}
        """.format(index.shape[0], pos_idx.shape[0], neg_idx.shape[0], round(pos_idx.shape[0]/index.shape[0]*100, 3),
                   round(f1_score(real_idx, pred_idx, average='weighted'), 4), ts_loss, ts_acc)
        print(desc)
        for idx in pos_idx:
            print('Positive Case:\t', TEST_DOC[index[idx]], '\t->\t', LABEL_IDX[np.argmax(proba[idx])],
                '({0:.2f})\t'.format(round(max(proba[idx]), 3)), LABEL_IDX[np.argmax(batch_label[idx])])

        for idx in neg_idx:

            print('Negative Case:\t', TEST_DOC[index[idx]], '\t->\t', LABEL_IDX[np.argmax(proba[idx])],
                  '({0:.2f})\t'.format(round(max(proba[idx]), 3)), LABEL_IDX[np.argmax(batch_label[idx])],
                  '({0:.2f})\t'.format(proba[idx][np.argmax(batch_label[idx])])
                  )

        print()

    if args.input == 'a.txt':
        testfile = open(filename)
        for line in testfile:
            line = line.rstrip('\n\r').lower()

            jaso_splitted = jmu.jaso_split([line], MAXLEN=FLAGS.INPUT_WIDTH)
            batch_input = sess.run(model.jaso_Onehot, {model.X_Onehot: jaso_splitted})
            y_proba = sess.run(model.y_proba, feed_dict={model.X: batch_input, model.TRAIN_PH: False})
            #print(batch_input.shape, y_proba.shape)
            label = LABEL_IDX[np.argmax(y_proba[0])]
            if round(max(y_proba[0]), 3) > 0.8:
                print(line, '\t->\t', label, round(max(y_proba[0]), 3))

    if args.input == 'tmp.txt':
        tmp_df = pd.read_csv(filename, sep='\t', header=None)
        tmp_df['class'] = tmp_df[1].apply(lambda x: json2intent(x))
        sentences = tmp_df[0].tolist()

        labels = []
        probas = []
        for sent in sentences:
            jaso_splitted = jmu.jaso_split([sent], MAXLEN=FLAGS.INPUT_WIDTH)
            batch_input = sess.run(model.jaso_Onehot, {model.X_Onehot: jaso_splitted})
            y_proba = sess.run(model.y_proba, feed_dict={model.X: batch_input, model.TRAIN_PH: False})
            # print(batch_input.shape, y_proba.shape)
            label = LABEL_IDX[np.argmax(y_proba[0])]
            labels.append(label)
            probas.append(round(max(y_proba[0]), 3))

        tmp_df['pred'] = labels
        tmp_df['proba'] = probas

        for index, row in tmp_df.iterrows():
            # if row['class'] is None:
            #     continue

            if row['class'] is not None:
                continue

            if round(row['proba'], 3) > 0.8:
                print(row[0], '\t->\t', row['pred'], round(row['proba'], 3), '\t', row['class'])

    if args.input == 'tmp_not_under.txt':
        tmp_df = pd.read_csv(filename, sep='\t', header=None)
        tmp_df['class'] = tmp_df[1].apply(lambda x: json2intent(x))
        sentences = tmp_df[0].tolist()

        labels = []
        probas = []
        for sent in sentences:
            jaso_splitted = jmu.jaso_split([sent], MAXLEN=FLAGS.INPUT_WIDTH)
            batch_input = sess.run(model.jaso_Onehot, {model.X_Onehot: jaso_splitted})
            y_proba = sess.run(model.y_proba, feed_dict={model.X: batch_input, model.TRAIN_PH: False})
            # print(batch_input.shape, y_proba.shape)
            label = LABEL_IDX[np.argmax(y_proba[0])]
            labels.append(label)
            probas.append(round(max(y_proba[0]), 3))

        tmp_df['pred'] = labels
        tmp_df['proba'] = probas

        for index, row in tmp_df.iterrows():

            if round(row['proba'], 3) > 0.1:
                print(row[1])
                print(row[0], '\t->\t', row['pred'], round(row['proba'], 3), '\t')
                print()


main()