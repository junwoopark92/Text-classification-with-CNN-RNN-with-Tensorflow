####################################################
# Data preprocessing
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
import re

def clean_str(sentence):

    sentence1 = re.sub(r"&.*?;", " ", sentence)
    sentence2 = re.findall(r"[a-zA-Z가-힣0-9]*", sentence1)
    res = ' '.join(sentence2)
    result = re.sub(r"\s{2,}", " ", res)

    return result.strip()

# #####################################
# # Preprocessing
# #####################################
# # The number of row: 6525185
# # Columns: movie review, point
# data = np.array(pd.read_csv('./Ch01_Data_load/data/w_movie.csv', encoding='cp949'))
# cleand_text = list(map(lambda x: clean_str(str(x)), data[:, 0]))
# data[:, 0] = cleand_text
#
# # Delete short sentences
# document_length = [len(str(x).split(" ")) for x in data[:, 0]]
# lb = int(np.percentile(document_length, 10))
# ub = int(np.percentile(document_length, 95))
# data = data[(np.array(document_length) > lb) & (np.array(document_length) < ub), :]
#
# # Delete mid-sentiment sentences
# # By Counter(_label), I determined cut value 2.5 (Negative : Positive ~ 2 : 3)
# del_point_tmp = np.r_[np.where((data[:, 1] > 2.5) & (data[:, 1] < 5))[0]]
# data = np.delete(data, del_point_tmp, axis=0)
#
# # Find unique sentences: 745493
# df = pd.DataFrame(data)
# unique_arr = df.drop_duplicates().values
#
# # Save numpy object
# np.save('./Ch01_Data_load/data/w_movie.npy', unique_arr)

##################################
# digi intent data preprocessing #
##################################


df = pd.read_csv('./data/mart_digi_speech_act.csv')
df['clean_text'] = df['sub_sent'].apply(lambda x: clean_str(str(x)))
df['len_text'] = df['clean_text'].apply(lambda x: len(x))
doc_len = df['len_text'].tolist()
lb = int(np.percentile(doc_len, 10))
ub = int(np.percentile(doc_len, 95))
df = df[(df['len_text'] > lb) & (df['len_text'] < ub)][['clean_text', 'class']]
df = df[['clean_text', 'class']]
#df = pd.concat([df[['clean_text']], pd.get_dummies(df['class'])], axis=1).reset_index(drop=True)
print(df.head())
df.to_csv('./data/preprocessed_mart_digi_speech_act.csv', index=False)
