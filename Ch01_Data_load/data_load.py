####################################################
# Data load
#  - Author: Deokseong Seo
#  - email: heyhi16@gmail.com
#  - git: https://github.com/DeokO
#  - Date: 2018.01.01.
####################################################



#####################################
# Import modules
#####################################
import numpy as np
import pandas as pd


#####################################
# Define functions
#####################################
# Float to one-hot
def label_to_onehot(label):
    # By Counter(_label), I determined cut value 2.5 (Negative : Positive ~ 2 : 3)
    _label = np.array(label > 2.5, dtype=np.int)
    output = np.zeros([len(_label), 2], dtype=np.float32)
    for ind in np.arange(len(_label)):
        output[ind, _label[ind]] = 1
    return output

# Data load

def data_load(onehot=True):
    data = np.load('./Ch01_Data_load/data/w_movie.npy')
    np.random.shuffle(data)
    tr_cut = np.round(len(data) * 0.9).astype(np.int32)
    TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL = data[:tr_cut, 0], data[:tr_cut, 1], data[tr_cut:, 0], data[tr_cut:, 1]

    # 작은 데이터로 OVERFITTING 시키기 위한 실험용
    # index = np.random.choice(range(len(TRAIN_DOC)), 300)
    # TRAIN_DOC=TRAIN_DOC[index]
    # TRAIN_LABEL=TRAIN_LABEL[index]
    # index = np.random.choice(range(len(TEST_DOC)), 300)
    # TEST_DOC = TEST_DOC[index]
    # TEST_LABEL = TEST_LABEL[index]

    TRAIN_LABEL_POS = np.where(label_to_onehot(TRAIN_LABEL)[:, 1] == 1)[0]
    TRAIN_LABEL_NEG = np.where(label_to_onehot(TRAIN_LABEL)[:, 1] == 0)[0]

    TEST_LABEL_POS = np.where(label_to_onehot(TEST_LABEL)[:, 1] == 1)[0]
    TEST_LABEL_NEG = np.where(label_to_onehot(TEST_LABEL)[:, 1] == 0)[0]

    if onehot:
        return TRAIN_DOC, label_to_onehot(TRAIN_LABEL), TRAIN_LABEL_POS, TRAIN_LABEL_NEG, TEST_DOC, label_to_onehot(TEST_LABEL), TEST_LABEL_POS, TEST_LABEL_NEG
    else:
        return TRAIN_DOC, TRAIN_LABEL, TRAIN_LABEL_POS, TRAIN_LABEL_NEG, TEST_DOC, TEST_LABEL, TEST_LABEL_POS, TEST_LABEL_NEG


def digi_data_load():
    df = pd.read_csv('/Users/1003874/bsdev/Text-classification-with-CNN-RNN-with-Tensorflow/Ch01_Data_load/data/preprocessed_digi_intent.csv')
    df['clean_text'] = df['clean_text'].apply(lambda x:x.lower())
    size = len(df)
    rate = 0.2
    train_df = df.iloc[:int(size*(1-rate))]
    test_df = df.iloc[int(size*(1-rate)):]
    label_df = pd.get_dummies(df['class'])
    print(len(train_df), len(test_df))

    TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL = \
        train_df['clean_text'].values, label_df.loc[train_df.index].values, test_df['clean_text'].values, label_df.loc[test_df.index].values

    return TRAIN_DOC, TRAIN_LABEL, TEST_DOC, TEST_LABEL