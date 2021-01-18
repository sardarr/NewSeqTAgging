"""
This code is to generate the the train, dev, and test set that we need for the deep learning experiments with all the features
This gets the pickle files that we tagged by the MTL and creates the appropriate format of file to be fed into the NN pipeline
The similar features enginieering as Logistic regresssion code used here.
"""


import argparse
import math
import string
from nltk import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import re
import pandas as pd
import numpy as np
import textstat
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset

import torch
from transformers import AutoModel, AutoTokenizer


from textblob import TextBlob
import os
from nltk.stem.snowball import SnowballStemmer
from sklearn import svm
import pickle
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
tf.random.set_seed(123)
np.random.seed(123)
from tensorflow.keras.layers import BatchNormalization

from Feature_extractor import clean_tweet, RetOrRep, urlChecker, top_url_returner, ligit_url_returner,sts


# count=0



def cm_analysis(y_true, y_pred, filename,labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="rocket_r")
    plt.savefig(filename)
    # plt.show()
    plt.close()


# Tokenize all of the sentences and map the tokens to thier word IDs.
def tokenizer_func(tokenizer_kind, sentences, labels):
    '''
    inputs:
      tokenizer_kind: is the the tokenizer of choice (normal bert, tweet bert)
      sentences: train , dev, test
    outputs:
    torchs of
      ids
      attention_mask
      labels
    '''
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer_kind.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=128,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True,
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels
    # Print sentence 0, now as a list of IDs.

def feature_calc(df,new_df,time,timeB):
    counter = 0
    df['retweetRate_frame' + str(time)] = ''
    df['replyRate_frame' + str(time)] = ''
    df['responseRate_frame' + str(time)] = ''
    df['commentRate_frame' + str(time)] = ''
    df['retweetRate_cum' + str(time)] = ''
    df['replyRate_cum' + str(time)] = ''
    df['responseRate_cum' + str(time)] = ''
    df['commentRate_cum' + str(time)] = ''

    df['max_depth_all' + str(time)] = ''
    df['mean_depth_all' + str(time)] = ''
    df['mean_depth_res_cum' + str(time)] = ''
    df['mean_depth_rep_cum' + str(time)] = ''
    df['mean_depth_ret_cum' + str(time)] = ''
    df['mean_depth_ret_frame' + str(time)] = ''
    df['mean_depth_rep_frame' + str(time)] = ''
    df['mean_depth_res_frame' + str(time)] = ''


    df['cb_all_cum' + str(time)] = ''
    df['ncb_all_cum' + str(time)] = ''
    df['na_all_cum' + str(time)] = ''
    df['rob_all_cum' + str(time)] = ''
    df['cb_frame' + str(time)] = ''
    df['ncb_frame' + str(time)] = ''
    df['na_frame' + str(time)] = ''
    df['rob_frame' + str(time)] = ''

    df['corse_word_frame' + str(time)] = ''
    df['corse_word_cum' + str(time)] = ''
    df['corse_word_cum_respons' + str(time)] = ''
    df['corse_word_cum_rep' + str(time)] = ''

    df['Sentiment_frame' + str(time)] = ''
    df['Sentiment_cum' + str(time)] = ''
    df['Sentiment_respons_cum' + str(time)] = ''
    df['Sentiment_rep_cum' + str(time)] = ''

    df['Sentiment_respons_frame' + str(time)] = ''
    df['Sentiment_rep_frame' + str(time)] = ''
    df['Sentiment_ret_frame' + str(time)] = ''

    df['Subjecitvity_frame' + str(time)] = ''
    df['Subjecitvity_cum' + str(time)] = ''
    df['Subjecitvity_respons_cum' + str(time)] = ''
    df['Subjecitvity_rep_cum' + str(time)] = ''

    df['Subjecitvity_respons_frame' + str(time)] = ''
    df['Subjecitvity_rep_frame' + str(time)] = ''
    df['Subjecitvity_ret_frame' + str(time)] = ''


    df['user_follower_res_frame' + str(time)] = ''
    df['user_follower_res_cum' + str(time)] = ''
    df['user_follower_rep_frame' + str(time)] = ''
    df['user_follower_rep_cum' + str(time)] = ''
    df['user_follower_ret_frame' + str(time)] = ''
    df['user_follower_ret_cum' + str(time)] = ''

    df['user_friends_count_res_frame' + str(time)] = ''
    df['user_friends_count_res_cum' + str(time)] = ''
    df['user_friends_count_rep_frame' + str(time)] = ''
    df['user_friends_count_rep_cum' + str(time)] = ''
    df['user_friends_count_ret_frame' + str(time)] = ''
    df['user_friends_count_ret_cum' + str(time)] = ''

    df['url_top_RT_frame' + str(time)] = ''
    df['url_top_res_frame' + str(time)] = ''
    df['url_top_rep_frame' + str(time)] = ''
    df['url_top_RT_cum' + str(time)] = ''
    df['url_top_Res_cum' + str(time)] = ''
    df['url_top_rep_cum' + str(time)] = ''


    df['url_ligit_RT_frame' + str(time)] = ''
    df['url_ligit_res_frame' + str(time)] = ''
    df['url_ligit_rep_frame' + str(time)] = ''
    df['url_ligit_RT_cum' + str(time)] = ''
    df['url_ligit_Res_cum' + str(time)] = ''
    df['url_ligit_rep_cum' + str(time)] = ''


    df['media_frame' + str(time)] = ''
    df['media_cum' + str(time)] = ''
    df['media_rep_frame' + str(time)] = ''
    df['media_rep_cum' + str(time)] = ''
    df['media_ret_frame' + str(time)] = ''
    df['media_ret_cum' + str(time)] = ''
    df['media_res_frame' + str(time)] = ''
    df['media_res_cum' + str(time)] = ''

    df['hash_frame' + str(time)] = ''
    df['hash_cum' + str(time)] = ''
    df['hash_rep_frame' + str(time)] = ''
    df['hash_rep_cum' + str(time)] = ''
    df['hash_ret_frame' + str(time)] = ''
    df['hash_ret_cum' + str(time)] = ''
    df['hash_res_frame' + str(time)] = ''
    df['hash_res_cum' + str(time)] = ''

    df['cb_frame' + str(time)] = ''
    df['cb_cum' + str(time)] = ''
    df['cb_respons_cum' + str(time)] = ''
    df['cb_rep_cum' + str(time)] = ''
    df['cb_ret_cum' + str(time)] = ''
    df['cb_respons_frame' + str(time)] = ''
    df['cb_rep_frame' + str(time)] = ''
    df['cb_ret_frame' + str(time)] = ''
    #
    df['ncb_frame' + str(time)] = ''
    df['ncb_cum' + str(time)] = ''
    df['ncb_respons_cum' + str(time)] = ''
    df['ncb_rep_cum' + str(time)] = ''
    df['ncb_ret_cum' + str(time)] = ''
    df['ncb_respons_frame' + str(time)] = ''
    df['ncb_rep_frame' + str(time)] = ''
    df['ncb_ret_frame' + str(time)] = ''
    #
    #
    df['rob_frame' + str(time)] = ''
    df['rob_cum' + str(time)] = ''
    df['rob_respons_cum' + str(time)] = ''
    df['rob_rep_cum' + str(time)] = ''
    df['rob_ret_cum' + str(time)] = ''
    df['rob_respons_frame' + str(time)] = ''
    df['rob_rep_frame' + str(time)] = ''
    df['rob_ret_frame' + str(time)] = ''

    df['na_frame' + str(time)] = ''
    df['na_cum' + str(time)] = ''
    df['na_respons_cum' + str(time)] = ''
    df['na_rep_cum' + str(time)] = ''
    df['na_ret_cum' + str(time)] = ''
    df['na_respons_frame' + str(time)] = ''
    df['na_rep_frame' + str(time)] = ''
    df['na_ret_frame' + str(time)] = ''

    df['stance_comment_frame' + str(time)] = ''
    df['stance_comment_cum' + str(time)] = ''
    df['stance_comment_respons_cum' + str(time)] = ''
    df['stance_comment_rep_cum' + str(time)] = ''
    df['stance_comment_ret_cum' + str(time)] = ''
    df['stance_comment_res_frame' + str(time)] = ''
    df['stance_comment_rep_frame' + str(time)] = ''
    df['stance_comment_ret_frame' + str(time)] = ''


    df['stance_query_frame' + str(time)] = ''
    df['stance_query_cum' + str(time)] = ''
    df['stance_query_respons_cum' + str(time)] = ''
    df['stance_query_rep_cum' + str(time)] = ''
    df['stance_query_ret_cum' + str(time)] = ''
    df['stance_query_res_frame' + str(time)] = ''
    df['stance_query_rep_frame' + str(time)] = ''
    df['stance_query_ret_frame' + str(time)] = ''


    df['stance_support_frame' + str(time)] = ''
    df['stance_support_cum' + str(time)] = ''
    df['stance_support_respons_cum' + str(time)] = ''
    df['stance_support_rep_cum' + str(time)] = ''
    df['stance_support_ret_cum' + str(time)] = ''
    df['stance_support_res_frame' + str(time)] = ''
    df['stance_support_rep_frame' + str(time)] = ''
    df['stance_support_ret_frame' + str(time)] = ''

    df['stance_deny_frame' + str(time)] = ''
    df['stance_deny_cum' + str(time)] = ''
    df['stance_deny_respons_cum' + str(time)] = ''
    df['stance_deny_rep_cum' + str(time)] = ''
    df['stance_deny_ret_cum' + str(time)] = ''
    df['stance_deny_res_frame' + str(time)] = ''
    df['stance_deny_rep_frame' + str(time)] = ''
    df['stance_deny_ret_frame' + str(time)] = ''

    for i, rows in df.iterrows():
        # if rows['src_rtw']=='src':
        if counter%500==0:
            print(counter)
        counter+=1
        df_time=new_df[(new_df['timeGap']<=time) & (new_df['timeGap'] > timeB) & (new_df['parent_ids']==str(rows['parent_ids']))]
        df_time_cum=new_df[(new_df['timeGap']<=time) & (new_df['timeGap'] > 0.0) & (new_df['parent_ids']==str(rows['parent_ids']))]

        df_time_RT=df_time_cum[df_time_cum['RetOrRep']==2]
        df_time_Rep=df_time_cum[df_time_cum['RetOrRep']==1]
        df_time_Res=df_time_cum[df_time_cum['RetOrRep']==0]

        df_time_RT_f=df_time[df_time['RetOrRep']==2]
        df_time_Rep_f=df_time[df_time['RetOrRep']==1]
        df_time_Res_f=df_time[df_time['RetOrRep']==0]

        # {'comment': 0, 'support': 1, 'deny': 2, 'query': 3, 'com ment': 0}

        df_time_stance_comment_frame=df_time[df_time['stance']==0]
        df_time_stance_comment_cum=df_time_cum[df_time_cum['stance']==0]

        df_time_stance_comment_frame_RT=[df_time_stance_comment_frame['RetOrRep']==2]
        df_time_stance_comment_frame_Rep=[df_time_stance_comment_frame['RetOrRep']==1]
        df_time_stance_comment_frame_Res=[df_time_stance_comment_frame['RetOrRep']==0]
        df_time_stance_comment_cum_RT=[df_time_stance_comment_cum['RetOrRep']==2]
        df_time_stance_comment_cum_Rep=[df_time_stance_comment_cum['RetOrRep']==1]
        df_time_stance_comment_cum_Res=[df_time_stance_comment_cum['RetOrRep']==0]

        df_time_stance_support_frame=df_time[df_time['stance']==1]
        df_time_stance_support_cum=df_time_cum[df_time_cum['stance']==1]

        df_time_stance_support_frame_RT=[df_time_stance_support_frame['RetOrRep']==2]
        df_time_stance_support_frame_Rep=[df_time_stance_support_frame['RetOrRep']==1]
        df_time_stance_support_frame_Res=[df_time_stance_support_frame['RetOrRep']==0]
        df_time_stance_support_cum_RT=[df_time_stance_support_cum['RetOrRep']==2]
        df_time_stance_support_cum_Rep=[df_time_stance_support_cum['RetOrRep']==1]
        df_time_stance_support_cum_Res=[df_time_stance_support_cum['RetOrRep']==0]



        df_time_stance_query_frame=df_time[df_time['stance']==3]
        df_time_stance_query_cum=df_time_cum[df_time_cum['stance']==3]

        df_time_stance_query_frame_RT=[df_time_stance_query_frame['RetOrRep']==2]
        df_time_stance_query_frame_Rep=[df_time_stance_query_frame['RetOrRep']==1]
        df_time_stance_query_frame_Res=[df_time_stance_query_frame['RetOrRep']==0]
        df_time_stance_query_cum_RT=[df_time_stance_query_cum['RetOrRep']==2]
        df_time_stance_query_cum_Rep=[df_time_stance_query_cum['RetOrRep']==1]
        df_time_stance_query_cum_Res=[df_time_stance_query_cum['RetOrRep']==0]


        df_time_stance_deny_frame=df_time[df_time['stance']==2]
        df_time_stance_deny_cum=df_time_cum[df_time_cum['stance']==2]

        df_time_stance_deny_frame_RT=[df_time_stance_deny_frame['RetOrRep']==2]
        df_time_stance_deny_frame_Rep=[df_time_stance_deny_frame['RetOrRep']==1]
        df_time_stance_deny_frame_Res=[df_time_stance_deny_frame['RetOrRep']==0]
        df_time_stance_deny_cum_RT=[df_time_stance_deny_cum['RetOrRep']==2]
        df_time_stance_deny_cum_Rep=[df_time_stance_deny_cum['RetOrRep']==1]
        df_time_stance_deny_cum_Res=[df_time_stance_deny_cum['RetOrRep']==0]


        df_time_RT_url_top_c = df_time_RT[df_time_RT['url_top'] != 501]
        df_time_url_top_Rep_c = df_time_Rep[df_time_Rep['url_top'] != 501]
        df_time_url_top_Res_c = df_time_Res[df_time_Res['url_top'] != 501]


        df_time_RT_url_top_f=df_time_RT_f[df_time_RT_f['url_top']!=501]
        df_time_url_top_Rep_f=df_time_Rep_f[df_time_Rep_f['url_top']!=501]
        df_time_url_top_Res_f=df_time_Res_f[df_time_Res_f['url_top']!=501]


        df_time_RT_url_ligit_c = df_time_RT[df_time_RT['url_ligit'] != 17]
        df_time_url_ligit_Rep_c = df_time_Rep[df_time_Rep['url_ligit'] != 17]
        df_time_url_ligit_Res_c = df_time_Res[df_time_Res['url_ligit'] != 17]

        df_time_RT_url_ligit_f=df_time_RT_f[df_time_RT_f['url_ligit']!=17]
        df_time_url_ligit_Rep_f=df_time_Rep_f[df_time_Rep_f['url_ligit']!=17]
        df_time_url_ligit_Res_f=df_time_Res_f[df_time_Res_f['url_ligit']!=17]


        ret = len(df_time_RT_f)
        rep = len(df_time_Rep_f)
        res = len(df_time_Res_f)
        comment = len(df_time)

        retC = len(df_time_RT)
        repC = len(df_time_Rep)
        resC = len(df_time_Res)
        commentC = len(df_time_cum)

        df.at[i, 'retweetRate_frame'+str(time)]=ret/(time*60)
        df.at[i, 'replyRate_frame'+str(time)] = rep / (time*60)
        df.at[i, 'responseRate_frame'+str(time)] = res / (time*60)
        df.at[i, 'commentRate_frame'+str(time)] = comment / (time*60)
        df.at[i, 'retweetRate_cum'+str(time)]=retC/(time*60)
        df.at[i, 'replyRate_cum'+str(time)] = repC / (time*60)
        df.at[i, 'responseRate_cum'+str(time)] = resC / (time*60)
        df.at[i, 'commentRate_cum'+str(time)] = commentC / (time*60)

        max_depth_all=df_time_cum['depth'].max()
        mean_depth_all=df_time_cum['depth'].mean()
        mean_depth_retC=df_time_RT['depth'].mean()
        mean_depth_repC=df_time_Rep['depth'].mean()
        mean_depth_resC=df_time_Res['depth'].mean()

        mean_depth_retf=df_time_RT_f['depth'].mean()
        mean_depth_repf=df_time_Rep_f['depth'].mean()
        mean_depth_resf=df_time_Res_f['depth'].mean()

        df.at[i, 'max_depth_all'+str(time)]=max_depth_all
        df.at[i, 'mean_depth_all'+str(time)]=mean_depth_all
        df.at[i, 'mean_depth_ret_cum'+str(time)] = mean_depth_retC
        df.at[i, 'mean_depth_rep_cum'+str(time)] = mean_depth_repC
        df.at[i, 'mean_depth_res_cum'+str(time)] = mean_depth_resC
        df.at[i, 'mean_depth_ret_frame'+str(time)] = mean_depth_retf
        df.at[i, 'mean_depth_rep_frame'+str(time)] = mean_depth_repf
        df.at[i, 'mean_depth_res_frame'+str(time)] = mean_depth_resf


        cb_all_cum=df_time_cum['cb_ev'].mean()
        ncb_all_cum=df_time_cum['ncb_ev'].mean()
        na_all_cum=df_time_cum['na_ev'].mean()
        rob_all_cum=df_time_cum['rob_ev'].mean()
        cb_frame=df_time['cb_ev'].mean()
        ncb_frame=df_time['ncb_ev'].mean()
        na_frame=df_time['na_ev'].mean()
        rob_frame=df_time['rob_ev'].mean()

        df.at[i, 'cb_all_cum'+str(time)]=cb_all_cum
        df.at[i, 'ncb_all_cum'+str(time)]=ncb_all_cum
        df.at[i, 'na_all_cum'+str(time)] = na_all_cum
        df.at[i, 'rob_all_cum'+str(time)] = rob_all_cum
        df.at[i, 'cb_frame'+str(time)] = cb_frame
        df.at[i, 'ncb_frame'+str(time)] = ncb_frame
        df.at[i, 'na_frame'+str(time)] = na_frame
        df.at[i, 'rob_frame'+str(time)] = rob_frame

        corse_word_frame = df_time['corse_word'].mean()
        corse_word_cum = df_time_cum['corse_word'].mean()
        corse_word_cum_respons=df_time_Res['corse_word'].mean()
        corse_word_cum_rep=df_time_Rep['corse_word'].mean()

        df.at[i, 'corse_word_frame'+str(time)] = corse_word_frame
        df.at[i, 'corse_word_cum'+str(time)] = corse_word_cum
        df.at[i, 'corse_word_cum_respons'+str(time)] = corse_word_cum_respons
        df.at[i, 'corse_word_cum_rep'+str(time)] = corse_word_cum_rep

        Sentiment_frame = df_time['Sentiment'].mean()
        Sentiment_cum = df_time_cum['Sentiment'].mean()
        Sentiment_respons_cum=df_time_Res['Sentiment'].mean()
        Sentiment_rep_cum=df_time_Rep['Sentiment'].mean()

        Sentiment_respons_frame=df_time_Res_f['Sentiment'].mean()
        Sentiment_rep_frame=df_time_Rep_f['Sentiment'].mean()
        Sentiment_ret_frame = df_time_RT_f['Sentiment'].mean()

        df.at[i, 'Sentiment_frame'+str(time)] = Sentiment_frame
        df.at[i, 'Sentiment_cum'+str(time)] = Sentiment_cum
        df.at[i, 'Sentiment_respons_cum'+str(time)] = Sentiment_respons_cum
        df.at[i, 'Sentiment_rep_cum'+str(time)] = Sentiment_rep_cum

        df.at[i, 'Sentiment_respons_frame' + str(time)] = Sentiment_respons_frame
        df.at[i, 'Sentiment_rep_frame' + str(time)] = Sentiment_rep_frame
        df.at[i, 'Sentiment_ret_frame' + str(time)] = Sentiment_ret_frame

        Subjecitvity_frame = df_time['Subjecitvity'].mean()
        Subjecitvity_cum = df_time_cum['Subjecitvity'].mean()
        Subjecitvity_respons_cum=df_time_Res['Subjecitvity'].mean()
        Subjecitvity_rep_cum=df_time_Rep['Subjecitvity'].mean()

        Subjecitvity_respons_frame=df_time_Res_f['Subjecitvity'].mean()
        Subjecitvity_rep_frame=df_time_Rep_f['Subjecitvity'].mean()
        Subjecitvity_ret_frame = df_time_RT_f['Subjecitvity'].mean()

        df.at[i, 'Subjecitvity_frame' + str(time)] = Subjecitvity_frame
        df.at[i, 'Subjecitvity_cum' + str(time)] = Subjecitvity_cum
        df.at[i, 'Subjecitvity_respons_cum' + str(time)] = Subjecitvity_respons_cum
        df.at[i, 'Subjecitvity_rep_cum' + str(time)] = Subjecitvity_rep_cum

        df.at[i, 'Subjecitvity_respons_frame' + str(time)] = Subjecitvity_respons_frame
        df.at[i, 'Subjecitvity_rep_frame' + str(time)] = Subjecitvity_rep_frame
        df.at[i, 'Subjecitvity_ret_frame' + str(time)] = Subjecitvity_ret_frame


        user_followers_RT_frame = df_time_RT_f['followers'].mean()
        user_followers_res_frame = df_time_Res_f['followers'].mean()
        user_followers_rep_frame = df_time_Rep_f['followers'].mean()
        user_followers_RT_cum = df_time_RT['followers'].mean()
        user_followers_Res_cum = df_time_Res['followers'].mean()
        user_followers_Rep_cum = df_time_Rep['followers'].mean()


        df.at[i, 'user_follower_res_frame' + str(time)] =user_followers_res_frame
        df.at[i, 'user_follower_res_cum' + str(time)] = user_followers_Res_cum
        df.at[i, 'user_follower_rep_frame' + str(time)] = user_followers_rep_frame
        df.at[i, 'user_follower_rep_cum' + str(time)] = user_followers_Rep_cum
        df.at[i, 'user_follower_ret_frame' + str(time)] = user_followers_RT_frame
        df.at[i, 'user_follower_ret_cum' + str(time)] = user_followers_RT_cum


        user_friends_count_RT_frame = df_time_RT_f['friends_count'].mean()
        user_friends_count_res_frame = df_time_Res_f['friends_count'].mean()
        user_friends_count_rep_frame = df_time_Rep_f['friends_count'].mean()
        user_friends_count_RT_cum = df_time_RT['friends_count'].mean()
        user_friends_count_Res_cum = df_time_Res['friends_count'].mean()
        user_friends_count_Rep_cum = df_time_Rep['friends_count'].mean()

        df.at[i, 'user_friends_count_res_frame' + str(time)] =user_friends_count_res_frame
        df.at[i, 'user_friends_count_res_cum' + str(time)] = user_friends_count_Res_cum
        df.at[i, 'user_friends_count_rep_frame' + str(time)] = user_friends_count_rep_frame
        df.at[i, 'user_friends_count_rep_cum' + str(time)] = user_friends_count_Rep_cum
        df.at[i, 'user_friends_count_ret_frame' + str(time)] = user_friends_count_RT_frame
        df.at[i, 'user_friends_count_ret_cum' + str(time)] = user_friends_count_RT_cum

        url_top_RT_frame = df_time_RT_url_top_f['url_top'].mode()
        if len(url_top_RT_frame)!=0:
            url_top_RT_frame=int(url_top_RT_frame.values[0])
        else:
            url_top_RT_frame = 501

        url_top_res_frame = df_time_url_top_Res_f['url_top'].mode()

        if len(url_top_res_frame)!=0:
            url_top_res_frame=int(url_top_res_frame.values[0])
        else:
            url_top_res_frame=501

        url_top_rep_frame = df_time_url_top_Rep_f['url_top'].mode()
        if len(url_top_rep_frame) != 0:
            url_top_rep_frame=int(url_top_rep_frame.values[0])
        else:
            url_top_rep_frame=501

        url_top_RT_cum = df_time_RT_url_top_c['url_top'].mode()
        if len(url_top_RT_cum)!=0:
            url_top_RT_cum=int(url_top_RT_cum.values[0])
        else:
            url_top_RT_cum=501


        url_top_Res_cum = df_time_url_top_Res_c['url_top'].mode()
        if len(url_top_Res_cum)!=0:
            url_top_Res_cum=int(url_top_Res_cum.values[0])
        else:
            url_top_Res_cum=501


        url_top_rep_cum = df_time_url_top_Rep_c['url_top'].mode()
        if len(url_top_rep_cum) != 0:
            url_top_rep_cum = int(url_top_rep_cum.values[0])
        else:
            url_top_rep_cum = 501


        df.at[i, 'url_top_RT_frame' + str(time)] =url_top_RT_frame
        df.at[i, 'url_top_res_frame' + str(time)] = url_top_res_frame
        df.at[i, 'url_top_rep_frame' + str(time)] = url_top_rep_frame
        df.at[i, 'url_top_RT_cum' + str(time)] = url_top_RT_cum
        df.at[i, 'url_top_Res_cum' + str(time)] = url_top_Res_cum
        df.at[i, 'url_top_rep_cum' + str(time)] = url_top_rep_cum
        #
        #
        url_ligit_RT_frame = df_time_RT_url_ligit_f['url_ligit'].mode()
        if len(url_ligit_RT_frame) != 0:
            url_ligit_RT_frame = int(url_ligit_RT_frame.values[0])
        else:
            url_ligit_RT_frame = 17

        url_ligit_res_frame = df_time_url_ligit_Res_f['url_ligit'].mode()
        if len(url_ligit_res_frame) != 0:
            url_ligit_res_frame = int(url_ligit_res_frame.values[0])
        else:
            url_ligit_res_frame = 17

        url_ligit_rep_frame = df_time_url_ligit_Rep_f['url_ligit'].mode()
        if len(url_ligit_rep_frame) != 0:
            url_ligit_rep_frame = int(url_ligit_rep_frame.values[0])
        else:
            url_ligit_rep_frame = 17

        url_ligit_RT_cum = df_time_RT_url_ligit_c['url_ligit'].mode()
        if len(url_ligit_RT_cum) != 0:
            url_ligit_RT_cum = int(url_ligit_RT_cum.values[0])
        else:
            url_ligit_RT_cum = 17

        url_ligit_Res_cum = df_time_url_ligit_Res_c['url_ligit'].mode()
        if len(url_ligit_Res_cum) != 0:
            url_ligit_Res_cum = int(url_ligit_Res_cum.values[0])
        else:
            url_ligit_Res_cum = 17

        url_ligit_rep_cum = df_time_url_ligit_Rep_c['url_ligit'].mode()
        if len(url_ligit_rep_cum) != 0:
            url_ligit_rep_cum = int(url_ligit_rep_cum.values[0])
        else:
            url_ligit_rep_cum = 17

        df.at[i, 'url_ligit_RT_frame' + str(time)] =url_ligit_RT_frame
        df.at[i, 'url_ligit_res_frame' + str(time)] = url_ligit_res_frame
        df.at[i, 'url_ligit_rep_frame' + str(time)] = url_ligit_rep_frame
        df.at[i, 'url_ligit_RT_cum' + str(time)] = url_ligit_RT_cum
        df.at[i, 'url_ligit_Res_cum' + str(time)] = url_ligit_Res_cum
        df.at[i, 'url_ligit_rep_cum' + str(time)] = url_ligit_rep_cum

        media_frame = df_time['media'].sum(axis=0)/(df_time['media'].size+1)
        media_cum = df_time_cum['media'].sum(axis=0)/(df_time_cum['media'].size+1)
        media_ret_frame = df_time_RT_f['media'].sum(axis=0)/(df_time_RT_f['media'].size+1)
        media_res_frame = df_time_Res_f['media'].sum(axis=0)/(df_time_Res_f['media'].size+1)
        media_rep_frame = df_time_Rep_f['media'].sum(axis=0)/(df_time_Rep_f['media'].size+1)
        media_RT_cum = df_time_RT['media'].sum(axis=0)/(df_time_RT['media'].size+1)
        media_Res_cum = df_time_Res['media'].sum(axis=0)/(df_time_Res['media'].size+1)
        media_Rep_cum = df_time_Rep['media'].sum(axis=0)/(df_time_Rep['media'].size+1)


        df.at[i, 'media_frame' + str(time)] = media_frame
        df.at[i, 'media_cum' + str(time)] = media_cum
        df.at[i, 'media_rep_frame' + str(time)] = media_rep_frame
        df.at[i, 'media_rep_cum' + str(time)] = media_Rep_cum
        df.at[i, 'media_ret_frame' + str(time)] = media_ret_frame
        df.at[i, 'media_ret_cum' + str(time)] = media_RT_cum
        df.at[i, 'media_res_frame' + str(time)] =media_res_frame
        df.at[i, 'media_res_cum' + str(time)] = media_Res_cum

        hash_frame = df_time['Hashtag'].sum(axis=0)/(df_time['Hashtag'].size+1)
        hash_cum = df_time_cum['Hashtag'].sum(axis=0)/(df_time_cum['Hashtag'].size+1)
        hash_rep_frame = df_time_RT_f['Hashtag'].sum(axis=0)/(df_time_RT_f['Hashtag'].size+1)
        hash_rep_cum = df_time_Res_f['Hashtag'].sum(axis=0)/(df_time_Res_f['Hashtag'].size+1)
        hash_ret_frame = df_time_Rep_f['Hashtag'].sum(axis=0)/(df_time_Rep_f['Hashtag'].size+1)
        hash_ret_cum = df_time_RT['Hashtag'].sum(axis=0)/(df_time_RT['Hashtag'].size+1)
        hash_res_frame = df_time_Res['Hashtag'].sum(axis=0)/(df_time_Res['Hashtag'].size+1)
        hash_res_cum = df_time_Rep['Hashtag'].sum(axis=0)/(df_time_Rep['Hashtag'].size+1)


        df.at[i, 'hash_frame' + str(time)] = hash_frame
        df.at[i, 'hash_cum' + str(time)] = hash_cum
        df.at[i, 'hash_rep_frame' + str(time)] = hash_rep_frame
        df.at[i, 'hash_rep_cum' + str(time)] = hash_rep_cum
        df.at[i, 'hash_ret_frame' + str(time)] = hash_ret_frame
        df.at[i, 'hash_ret_cum' + str(time)] = hash_ret_cum
        df.at[i, 'hash_res_frame' + str(time)] =hash_res_frame
        df.at[i, 'hash_res_cum' + str(time)] = hash_res_cum
    #
        cb_frame = df_time['cb_ev'].mean()
        cb_cum = df_time_cum['cb_ev'].mean()
        cb_respons_cum = df_time_Res['cb_ev'].mean()
        cb_rep_cum = df_time_Rep['cb_ev'].mean()
        cb_ret_cum = df_time_RT['cb_ev'].mean()
        cb_respons_frame = df_time_Res_f['cb_ev'].mean()
        cb_rep_frame = df_time_Rep_f['cb_ev'].mean()
        cb_ret_frame = df_time_RT_f['cb_ev'].mean()

        df.at[i, 'cb_frame' + str(time)] = cb_frame
        df.at[i, 'cb_cum' + str(time)] = cb_cum
        df.at[i, 'cb_respons_cum' + str(time)] = cb_respons_cum
        df.at[i, 'cb_rep_cum' + str(time)] = cb_rep_cum
        df.at[i, 'cb_ret_cum' + str(time)] = cb_ret_cum
        df.at[i, 'cb_respons_frame' + str(time)] = cb_respons_frame
        df.at[i, 'cb_rep_frame' + str(time)] =cb_rep_frame
        df.at[i, 'cb_ret_frame' + str(time)] = cb_ret_frame

        rob_frame = df_time['rob_ev'].mean()
        rob_cum = df_time_cum['rob_ev'].mean()
        rob_respons_cum = df_time_Res['rob_ev'].mean()
        rob_rep_cum = df_time_Rep['rob_ev'].mean()
        rob_ret_cum = df_time_RT['rob_ev'].mean()
        rob_respons_frame = df_time_Res_f['rob_ev'].mean()
        rob_rep_frame = df_time_Rep_f['rob_ev'].mean()
        rob_ret_frame = df_time_RT_f['rob_ev'].mean()

        df.at[i, 'rob_frame' + str(time)] = rob_frame
        df.at[i, 'rob_cum' + str(time)] = rob_cum
        df.at[i, 'rob_respons_cum' + str(time)] = rob_respons_cum
        df.at[i, 'rob_rep_cum' + str(time)] = rob_rep_cum
        df.at[i, 'rob_ret_cum' + str(time)] = rob_ret_cum
        df.at[i, 'rob_respons_frame' + str(time)] = rob_respons_frame
        df.at[i, 'rob_rep_frame' + str(time)] = rob_rep_frame
        df.at[i, 'rob_ret_frame' + str(time)] = rob_ret_frame

        ncb_frame = df_time['ncb_ev'].mean()
        ncb_cum = df_time_cum['ncb_ev'].mean()
        ncb_respons_cum = df_time_Res['ncb_ev'].mean()
        ncb_rep_cum = df_time_Rep['ncb_ev'].mean()
        ncb_ret_cum = df_time_RT['ncb_ev'].mean()
        ncb_respons_frame = df_time_Res_f['ncb_ev'].mean()
        ncb_rep_frame = df_time_Rep_f['ncb_ev'].mean()
        ncb_ret_frame = df_time_RT_f['ncb_ev'].mean()

        df.at[i, 'ncb_frame' + str(time)] = ncb_frame
        df.at[i, 'ncb_cum' + str(time)] = ncb_cum
        df.at[i, 'ncb_respons_cum' + str(time)] = ncb_respons_cum
        df.at[i, 'ncb_rep_cum' + str(time)] = ncb_rep_cum
        df.at[i, 'ncb_ret_cum' + str(time)] = ncb_ret_cum
        df.at[i, 'ncb_respons_frame' + str(time)] = ncb_respons_frame
        df.at[i, 'ncb_rep_frame' + str(time)] = ncb_rep_frame
        df.at[i, 'ncb_ret_frame' + str(time)] = ncb_ret_frame

        na_frame = df_time['na_ev'].mean()
        na_cum = df_time_cum['na_ev'].mean()
        na_respons_cum = df_time_Res['na_ev'].mean()
        na_rep_cum = df_time_Rep['na_ev'].mean()
        na_ret_cum = df_time_RT['na_ev'].mean()
        na_respons_frame = df_time_Res_f['na_ev'].mean()
        na_rep_frame = df_time_Rep_f['na_ev'].mean()
        na_ret_frame = df_time_RT_f['na_ev'].mean()

        df.at[i, 'na_frame' + str(time)] = na_frame
        df.at[i, 'na_cum' + str(time)] = na_cum
        df.at[i, 'na_respons_cum' + str(time)] = na_respons_cum
        df.at[i, 'na_rep_cum' + str(time)] = na_rep_cum
        df.at[i, 'na_ret_cum' + str(time)] = na_ret_cum
        df.at[i, 'na_respons_frame' + str(time)] = na_respons_frame
        df.at[i, 'na_rep_frame' + str(time)] = na_rep_frame
        df.at[i, 'na_ret_frame' + str(time)] = na_ret_frame


        stance_comment_frame = len(df_time_stance_comment_frame.index)/(len(df_time.index)+1)
        stance_comment_cum = len(df_time_stance_comment_cum.index)/(len(df_time_cum.index)+1)
        stance_comment_respons_cum=len(df_time_stance_comment_cum_Res)/(len(df_time_Res)+1)
        stance_comment_rep_cum=len(df_time_stance_comment_cum_Rep)/(len(df_time_Rep)+1)
        stance_comment_ret_cum=len(df_time_stance_comment_cum_RT)/(len(df_time_RT)+1)
        stance_comment_res_frame=len(df_time_stance_comment_frame_Res)/(len(df_time_Res_f)+1)
        stance_comment_rep_frame=len(df_time_stance_comment_frame_Rep)/(len(df_time_Rep_f)+1)
        stance_comment_ret_frame=len(df_time_stance_comment_frame_RT)/(len(df_time_RT_f)+1)

        df.at[i, 'stance_comment_frame' + str(time)] = stance_comment_frame
        df.at[i, 'stance_comment_cum' + str(time)] = stance_comment_cum
        df.at[i, 'stance_comment_respons_cum' + str(time)] = stance_comment_respons_cum
        df.at[i, 'stance_comment_rep_cum' + str(time)] = stance_comment_rep_cum
        df.at[i, 'stance_comment_ret_cum' + str(time)] = stance_comment_ret_cum
        df.at[i, 'stance_comment_res_frame' + str(time)] = stance_comment_res_frame
        df.at[i, 'stance_comment_rep_frame' + str(time)] = stance_comment_rep_frame
        df.at[i, 'stance_comment_ret_frame' + str(time)] = stance_comment_ret_frame


        stance_query_frame = len(df_time_stance_query_frame.index)/(len(df_time.index)+1)
        stance_query_cum = len(df_time_stance_query_cum.index)/(len(df_time_cum.index)+1)
        stance_query_respons_cum=len(df_time_stance_query_cum_Res)/(len(df_time_Res)+1)
        stance_query_rep_cum=len(df_time_stance_query_cum_Rep)/(len(df_time_Rep)+1)
        stance_query_ret_cum=len(df_time_stance_query_cum_RT)/(len(df_time_RT)+1)
        stance_query_res_frame=len(df_time_stance_query_frame_Res)/(len(df_time_Res_f)+1)
        stance_query_rep_frame=len(df_time_stance_query_frame_Rep)/(len(df_time_Rep_f)+1)
        stance_query_ret_frame=len(df_time_stance_query_frame_RT)/(len(df_time_RT_f)+1)

        df.at[i, 'stance_query_frame' + str(time)] = stance_query_frame
        df.at[i, 'stance_query_cum' + str(time)] = stance_query_cum
        df.at[i, 'stance_query_respons_cum' + str(time)] = stance_query_respons_cum
        df.at[i, 'stance_query_rep_cum' + str(time)] = stance_query_rep_cum
        df.at[i, 'stance_query_ret_cum' + str(time)] = stance_query_ret_cum
        df.at[i, 'stance_query_res_frame' + str(time)] = stance_query_res_frame
        df.at[i, 'stance_query_rep_frame' + str(time)] = stance_query_rep_frame
        df.at[i, 'stance_query_ret_frame' + str(time)] = stance_query_ret_frame


        stance_support_frame = len(df_time_stance_support_frame.index)/(len(df_time.index)+1)
        stance_support_cum = len(df_time_stance_support_cum.index)/(len(df_time_cum.index)+1)
        stance_support_respons_cum=len(df_time_stance_support_cum_Res)/(len(df_time_Res)+1)
        stance_support_rep_cum=len(df_time_stance_support_cum_Rep)/(len(df_time_Rep)+1)
        stance_support_ret_cum=len(df_time_stance_support_cum_RT)/(len(df_time_RT)+1)
        stance_support_res_frame=len(df_time_stance_support_frame_Res)/(len(df_time_Res_f)+1)
        stance_support_rep_frame=len(df_time_stance_support_frame_Rep)/(len(df_time_Rep_f)+1)
        stance_support_ret_frame=len(df_time_stance_support_frame_RT)/(len(df_time_RT_f)+1)

        df.at[i, 'stance_support_frame' + str(time)] = stance_support_frame
        df.at[i, 'stance_support_cum' + str(time)] = stance_support_cum
        df.at[i, 'stance_support_respons_cum' + str(time)] = stance_support_respons_cum
        df.at[i, 'stance_support_rep_cum' + str(time)] = stance_support_rep_cum
        df.at[i, 'stance_support_ret_cum' + str(time)] = stance_support_ret_cum
        df.at[i, 'stance_support_res_frame' + str(time)] = stance_support_res_frame
        df.at[i, 'stance_support_rep_frame' + str(time)] = stance_support_rep_frame
        df.at[i, 'stance_support_ret_frame' + str(time)] = stance_support_ret_frame

        stance_deny_frame = len(df_time_stance_deny_frame.index)/(len(df_time.index)+1)
        stance_deny_cum = len(df_time_stance_deny_cum.index)/(len(df_time_cum.index)+1)
        stance_deny_respons_cum = len(df_time_stance_deny_cum_Res) / (len(df_time_Res) + 1)
        stance_deny_rep_cum = len(df_time_stance_deny_cum_Rep) / (len(df_time_Rep) + 1)
        stance_deny_ret_cum = len(df_time_stance_deny_cum_RT) / (len(df_time_RT) + 1)
        stance_deny_res_frame = len(df_time_stance_deny_frame_Res) / (len(df_time_Res_f) + 1)
        stance_deny_rep_frame = len(df_time_stance_deny_frame_Rep) / (len(df_time_Rep_f) + 1)
        stance_deny_ret_frame = len(df_time_stance_deny_frame_RT) / (len(df_time_RT_f) + 1)


        df.at[i, 'stance_deny_frame' + str(time)] = stance_deny_frame
        df.at[i, 'stance_deny_cum' + str(time)] = stance_deny_cum
        df.at[i, 'stance_deny_respons_cum' + str(time)] = stance_deny_respons_cum
        df.at[i, 'stance_deny_rep_cum' + str(time)] = stance_deny_rep_cum
        df.at[i, 'stance_deny_ret_cum' + str(time)] = stance_deny_ret_cum
        df.at[i, 'stance_deny_res_frame' + str(time)] = stance_deny_res_frame
        df.at[i, 'stance_deny_rep_frame' + str(time)] = stance_deny_rep_frame
        df.at[i, 'stance_deny_ret_frame' + str(time)] = stance_deny_ret_frame


print("Just test")

cleanup_nums = {'stance': {'comment': 0, 'support': 1, 'deny': 2,'query':3,'com ment':0},'src_rtw':{'ret':0, 'src':1}}

def featureGen(data,dstype,MODE):
    count=0
    if MODE == "Load":
        corse_words = set([cw[:-1] for cw in open("curse.txt", 'r').readlines()])
        data['corse_word'] = ''
        data['sylCount'] = ''
        data['reading_ease'] = ''
        data['reading_ease_parent'] = ''
        data['Sentiment'] = ''
        data['Subjecitvity'] = ''
        data['Sentiment_parent'] = ''
        data['Subjecitvity_parent'] = ''
        data['RetOrRep_parent'] = ''
        data['url_top']=''
        data['url_ligit']=''
        data['sts']=''
        # data['Botometer_score_english'] = ''
        # data['Botometer_score_unversal'] = ''
        for i, rows in data.iterrows():
            clearn_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",rows['text']).split())
            clearn_text_parent = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",rows['parent_body']).split())
            data.at[i, 'RetOrRep'] = RetOrRep(rows['text'])
            data.at[i, 'RetOrRep_parent'] = RetOrRep(rows['parent_body'])
            data.at[i, 'sts'] = sts(clearn_text, clearn_text_parent)
            data.at[i, 'corse_word'] = len(["true" for w in clearn_text.split() if w in corse_words])
            data.at[i, 'sylCount'] = textstat.syllable_count(clearn_text, lang='en_US')
            data.at[i, 'reading_ease_parent'] = textstat.automated_readability_index(clearn_text_parent)
            data.at[i, 'reading_ease'] = textstat.automated_readability_index(clearn_text)
            textBlob = TextBlob(clean_tweet(clearn_text))
            data.at[i, 'Sentiment'] = textBlob.sentiment.polarity
            data.at[i, 'Subjecitvity'] = textBlob.sentiment.subjectivity
            data = data[data.subnode_max_depth != 10000]
            textBlob_parent = TextBlob(clean_tweet(clearn_text_parent))
            data.at[i, 'Sentiment_parent'] = textBlob_parent.sentiment.polarity
            data.at[i, 'Subjecitvity_parent'] = textBlob_parent.sentiment.subjectivity
            if count % 500 == 0:
                print(count),
            count += 1


        pd.to_pickle(data,'data/Rumor/Logistic_regression_ver'+dstype)
        X_org = pd.DataFrame(data, columns=['ups', 'text', 'parent_body','num_comment','story', 'rum_tag','src_rtw','id','sts' ,'stance','sylCount', 'reading_ease', 'reading_ease_parent', \
                                        'url', 'Hashtag', 'Sentiment', 'Subjecitvity', 'corse_word', 'RetOrRep_parent',
                                        'RetOrRep', 'verified','depth','parent_ids',\
                                        'media', 'length', 'followers', 'timeGap', 'Sentiment_parent', 'user_description', \
                                        'listed_count', 'veracitytag', 'subnode_max_depth', 'statuses_count','platform',
                                        'Subjecitvity_parent', 'friends_count', 'user_url', \
                                        'cb_ev', 'ncb_ev', 'na_ev', 'rob_ev','url_ligit','url_top'])  # independent columns
    else:
        data = pd.read_pickle("data/Rumor/Logistic_regression_ver"+dstype)
        data=data[data.subnode_max_depth != 10000]
        data['url_top']=''
        data['url_ligit']=''
        X_org = pd.DataFrame(data, columns=['ups','text','num_comment','story','parent_body','created','id','src_rtw','sts','stance','rum_tag','sylCount','reading_ease','reading_ease_parent',\
                    'url','Hashtag','Sentiment','Subjecitvity','corse_word','RetOrRep_parent','RetOrRep','verified',\
                    'media','length','followers','timeGap','Sentiment_parent','depth','parent_ids','user_description','platform',\
                    'listed_count','veracitytag','subnode_max_depth', 'statuses_count','Subjecitvity_parent','friends_count', 'user_url',\
                    'cb_ev','ncb_ev', 'na_ev','rob_ev','url_ligit','url_top'])  #independent columns

    URL_TOP_PATH = 'URLs/top500.csv'
    URL_LIGIT_PATH = 'URLs/sources.csv'


    top_url,ligit_url=urlChecker(URL_TOP_PATH,URL_LIGIT_PATH)

    print("The main pickle file is built......")

    X_org['url_top'] = X_org['url'].map(lambda a: top_url_returner(a,top_url) if len(a)>0 else 501)
    X_org['url_ligit'] = X_org['url'].map(lambda a: ligit_url_returner(a,ligit_url) if len(a)>0 else 17)
    X_org['url'] = np.where(X_org.astype(str)['url'] != '[]' , 1,0)
    X_org['verified'] = np.where(X_org['verified']=='True' , 1,0)
    X_org['Hashtag'] = np.where(X_org.astype(str)['Hashtag'] != '[]' , 1,0)
    X_org['media'] = np.where(X_org.astype(str)['media'] != '' , 1,0)
    X_org['user_description'] = np.where(X_org.astype(str)['user_description'] != '' , 1,0)
    X_org['user_url'] = np.where(X_org.astype(str)['user_url']!='None' , 1,0)
    # X['veracitytag'] = np.where(X['veracitytag']=='1' , 1,0)
    X_org['platform'] = np.where(X_org.astype(str)['platform']!='twitter' , 1,0)

    # Converting some columns to categorical by casting the to string
    X_org.fillna(X_org.mean(), inplace=True)
    X_org['followers'] = X_org.followers.astype(int)
    X_org['friends_count'] = X_org.friends_count.astype(int)
    X_org['listed_count'] = X_org.listed_count.astype(int)
    X_org['statuses_count'] = X_org.statuses_count.astype(int)

    X_org=X_org.sample(frac=1)

            # 'veracitytag':{'nVer':3,'1':1,'0':0}}


    X_org.loc[X_org['friends_count'] == 0, 'friends_count'] = 1  # converting people with zero friends to one
    X_org['friends_count'] = X_org['friends_count'].map(lambda a: math.log10(a))
    X_org.loc[X_org['followers'] == 0, 'followers'] = 1  # converting people with zero friends to one
    X_org['followers'] = X_org['followers'].map(lambda a: math.log10(a))
    X_org.loc[X_org['listed_count'] == 0, 'listed_count'] = 1  # converting people with zero friends to one
    X_org['listed_count'] = X_org['listed_count'].map(lambda a: math.log10(a))
    X_org.loc[X_org['statuses_count'] == 0, 'statuses_count'] = 1  # converting people with zero friends to one
    X_org['statuses_count'] = X_org['statuses_count'].map(lambda a: math.log10(a))

    X = X_org

    X.replace(cleanup_nums, inplace=True)



    return X


#filtering section
# X=X.loc[X['timeGap'] == 0]



 # Time gaps for the tests 1 means one hour 0.1 mean 6 minutes

# sns.catplot(x="timeGap",y="story",kind='violin',data=X_org)


def time_frame_feature_generatro(X_train='none',dstype='train',time=6):
    """
    Args:
        X_train: dataframe which contains the whole dataset
        dstype: train,test,or dev
        time: extracting the features up to that specific time frame

    Returns: saving the dataset with extracted features and returning the dataframe in specified time frame by defult is 6 which means the largest df with all the features which is extracted from
    the whole time gaps
    """
    cleanup_numstag = {'veracitytag':{'PARfalse':4, 'PARtrue':3, 'PARunverified':5, 'false':0, 'unverified':2,
       'true':1}}
    if args.tasks[0] == 'ver' and not args.vern:
        cleanup_numstag = {'veracitytag': {'PARfalse': 0, 'PARtrue': 1, 'PARunverified': 2, 'false': 0, 'unverified': 2,
                                           'true': 1}}
    TIME_GAP = [0, 0.1, 0.33, 1.2, 72, 168, 700]
    X = X_train
    if os.path.isfile('data/Rumor/ver_stance_features_' + str(time) + dstype):
        X = pd.read_pickle('data/Rumor/ver_stance_features_' + str(time) + dstype)
        X.replace(cleanup_numstag, inplace=True)
        return X

    for ind,gaps in enumerate(TIME_GAP):
            # [x for x in list(os_data_X.columns) if len(os_data_X[x].unique()) == 1]
        if gaps != 0:
            timeBegin=0
            if ind!=0:
                timeBegin=TIME_GAP[ind - 1]
            if os.path.isfile('data/Rumor/ver_stance_features_' + str(ind) + dstype):
                X = pd.read_pickle('data/Rumor/ver_stance_features_' + str(ind) + dstype)
            else:
                new_X = X_train[X_train['timeGap'] <= gaps]
                feature_calc(X,new_X,gaps,timeBegin)
                X.fillna(X.mean(), inplace=True)
                pd.to_pickle(X, 'data/Rumor/ver_stance_features_'+str(ind)+dstype)
                print(X.shape)
        else:
            if os.path.isfile('data/Rumor/ver_stance_features_'+str(ind)+dstype):
                X=pd.read_pickle('data/Rumor/ver_stance_features_'+str(ind)+dstype)
            else:
                X.fillna(X.mean(), inplace=True)
                pd.to_pickle(X, 'data/Rumor/ver_stance_features_'+str(ind)+dstype)
        if ind==time:
            X.replace(cleanup_numstag, inplace=True)
            return X

def upsampler(datframe, task, srcOnly, upsamling=True):
    if task == 'ver' and srcOnly=='srcOnly' and upsamling:
        df_true = datframe[datframe.veracitytag == 1]  # Rum
        df_false = datframe[datframe.veracitytag == 0]
        df_unver = datframe[datframe.veracitytag == 2]
        minor_unver = resample(df_unver, replace=True,  # sample with replacement
                                         n_samples=144,  # to match majority class
                                         random_state=123)
        minor_false = resample(df_false,replace=True,  # sample with replacement
                                         n_samples=144,  # to match majority class
                                         random_state=123)
        df_upsampled = pd.concat([minor_unver, minor_false, df_true])

    elif task == 'ver' and upsamling and not args.vern:
        df_true = datframe[datframe.veracitytag == 1]
        df_false = datframe[datframe.veracitytag == 0]
        df_unver = datframe[datframe.veracitytag == 2]
        minor_unver = resample(df_unver, replace=True,  # sample with replacement
                               n_samples = df_true.shape[0],  # to match majority class
                               random_state=123)
        minor_false = resample(df_false, replace=True,  # sample with replacement
                               n_samples = df_true.shape[0],  # to match majority class
                               random_state=123)
        df_upsampled = pd.concat([minor_unver, minor_false, df_true])
        # df_upsampled = datframe

    elif task == 'ver' and upsamling:
        df_true = datframe[datframe.veracitytag == 1]
        df_false = datframe[datframe.veracitytag == 0]
        df_unver = datframe[datframe.veracitytag == 2]
        df_par_true = datframe[datframe.veracitytag == 3]
        df_par_false = datframe[datframe.veracitytag == 4]
        df_par_unver = datframe[datframe.veracitytag == 5]
        minor_unver = resample(df_unver, replace=True,  # sample with replacement
                               n_samples=400,  # to match majority class
                               random_state=123)
        minor_false = resample(df_false, replace=True,  # sample with replacement
                               n_samples=400,  # to match majority class
                               random_state=123)
        df_true = resample(df_true, replace=True,  # sample with replacement
                               n_samples=400,  # to match majority class
                               random_state=123)
        df_par_true = resample(df_par_true, replace=True,  # sample with replacement
                               n_samples=400,  # to match majority class
                               random_state=123)
        df_par_false = resample(df_par_false, replace=True,  # sample with replacement
                               n_samples=400,  # to match majority class
                               random_state=123)
        df_par_unver = resample(df_par_unver, replace=True,  # sample with replacement
                               n_samples=400,  # to match majority class
                               random_state=123)
        df_upsampled = pd.concat([minor_unver, minor_false, df_true,df_par_true,df_par_false,df_par_unver])

    elif task == 'stance' and upsamling:
        df_comment = datframe[datframe.stance == 0]  # Rum
        df_support = datframe[datframe.stance == 1]
        df_deny = datframe[datframe.stance == 2]
        df_query = datframe[datframe.stance == 3]
        df_comment = resample(df_comment, replace=True,  # sample with replacement
                               n_samples=1000,  # to match majority class
                               random_state=123)
        df_support = resample(df_support, replace=True,  # sample with replacement
                              n_samples=1000,  # to match majority class
                              random_state=123)
        df_deny = resample(df_deny, replace=True,  # sample with replacement
                              n_samples=1000,  # to match majority class
                              random_state=123)
        df_query = resample(df_query, replace=True,  # sample with replacement
                              n_samples=1000,  # to match majority class
                              random_state=123)
        df_upsampled = pd.concat([df_comment, df_support, df_deny,df_query])

    if task =='ver' and upsamling==False:
        df_upsampled=datframe
        df_upsampled.drop(columns=['stance'], inplace=True)
    elif task == 'stance' and upsamling==False:
        df_upsampled=datframe

    return df_upsampled.sample(frac=1).reset_index(drop=True)

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_nonprintable(text):
    """Remove non printable characters from a string"""
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))

def dl_data_prep(labels):
    temp_lables = []
    for label in labels:
        dummy = [0, 0, 0]
        dummy[label] = 1
        temp_lables.append(dummy)
    return np.array(temp_lables)

pattern = r'\w+|\?|\!|\"|\'|\;|\:'
class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(pattern)
        self.stemmer = stemmer
    def __call__(self, doc):
        return [self.stemmer.stem(token)
                for token in self.tok.tokenize(doc)]

stemmer = SnowballStemmer("english")


#I decided to have another method her for logging each test including ver,ver6, and stance seperatly
# becuase we have to run the same test for dev and test in that is redundant

from sklearn.metrics import f1_score
def f1_h(y_true,y_pred):
  y_true_int=np.argmax(y_true,axis=1)
  y_pred_int=np.argmax(y_pred,axis=1)
  macro=f1_score(y_true_int, y_pred_int, average='macro')
  print(classification_report(y_true_int, y_pred_int, target_names = ['False', 'True', 'Unverified']))
  print(confusion_matrix(y_true_int, y_pred_int))
  return macro


def train_test(train, dev, test, arg, settging, logver, logStance, logver6, logver6n):
    task=settging[1]
    if task =='stance':
        if settging[2] =='srcOnly' or settging[3]!=6 or settging[5]!='all' :
            return

    models = {'svm': svm.SVC(gamma='auto'), 'Rf': RandomForestClassifier(n_estimators=100, max_depth=20), \
              'NB': BernoulliNB(), 'ADA': AdaBoostClassifier()}
    #[platform,task,sns,ind,k,tw_typ,framecategory]
    # train = train[~train.index.duplicated()]
    # dev = dev[~dev.index.duplicated()]
    # test = test[~test.index.duplicated()]

    train = train.loc[:, ~train.columns.duplicated()]
    dev = dev.loc[:, ~dev.columns.duplicated()]
    test = test.loc[:, ~test.columns.duplicated()]

    train.fillna(train.mean(), inplace=True)
    dev.fillna(train.mean(), inplace=True)
    test.fillna(train.mean(), inplace=True)

    if args.dldata:
        pd.to_pickle(train,'data/Rumor/train_ver_stance_nn_'+str(settging[3]))
        pd.to_pickle(dev,'data/Rumor/dev_ver_stance_nn_'+str(settging[3]))
        pd.to_pickle(test,'data/Rumor/test_ver_stance_nn_'+str(settging[3]))
        print("Making the files for deep learning pipeline is done...")
        # quit()



    dev_twitter = dev[dev['platform']==0]
    dev_reddit = dev[dev['platform']==1]
    y_dev_twitter = dev_twitter['veracitytag']
    y_dev_reddit = dev_reddit['veracitytag']

    test_twitter = test[test['platform']==0]
    test_reddit = test[test['platform']==1]
    y_test_twitter = test_twitter['veracitytag']
    y_test_reddit = test_reddit['veracitytag']

    X_train_upsampled = upsampler(train, settging[1], settging[2], True)
    y_train_upsampled = X_train_upsampled['veracitytag']
    y_dev = dev['veracitytag']
    y_test = test['veracitytag']
    excluded_features = ['id', 'story', 'num_comment', 'rum_tag', 'veracitytag', 'created',
                         'RetOrRep_parent', 'parent_ids', 'subnode_max_depth', 'depth','parent_body', 'timeGap', \
                          'ups']


    if task =='stance':
        y_train_upsampled = X_train_upsampled['stance']
        y_dev = dev['stance']
        y_test = test['stance']
        excluded_features = ['id', 'story', 'num_comment', 'rum_tag', 'veracitytag','stance','parent_body', 'created',
                             'RetOrRep_parent', 'parent_ids', 'subnode_max_depth', 'depth', 'timeGap', \
                             'ups']
        y_dev_twitter = dev_twitter['stance']
        y_dev_reddit = dev_reddit['stance']
        y_test_twitter = test_twitter['stance']
        y_test_reddit = test_reddit['stance']


    test_set_container = {'dev':dev,'test':test,'dev_twitter':dev_twitter,'dev_reddit':dev_reddit,'test_twitter':test_twitter,'test_reddit':test_reddit}
    y_test_set_container = {'y_dev':y_dev,'y_test':y_test,'y_dev_twitter':y_dev_twitter,'y_dev_reddit':y_dev_reddit,'y_test_twitter':y_test_twitter,'y_test_reddit':y_test_reddit}

    X = X_train_upsampled[X_train_upsampled.columns.difference(excluded_features)]
    # pca = PCA().fit(X)
    # # Plotting the Cumulative Summation of the Explained Variance
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Variance (%)')  # for each component
    # plt.title('Pulsar Dataset Explained Variance')
    # plt.show()
    try:
        X.replace(cleanup_nums, inplace=True)
    except:
        print("You dont need to replace")

    if arg.model == 'transformer':
        # train_x, test_x, dev_x, train_y, test_y, dev_y = X_train_upsampled ,

        train = X_train_upsampled[X_train_upsampled['src_rtw'] == 1]
        dev = dev[dev['src_rtw'] == 1]
        test = test[test['src_rtw'] == 1]

        ver_tag_ind = {'veracitytag': {'PARfalse': 4, 'PARtrue': 3, 'PARunverified': 5, 'false': 0, 'unverified': 2,
                                       'true': 1}}



        train_sentences = train.text.values
        train_labels = train["veracitytag"].values
        test_sentences = test.text.values
        test_labels = test["veracitytag"].values
        dev_sentences = dev.text.values
        dev_labels = dev["veracitytag"].values

        excluded_features = ['id', 'parent_body', 'parent_ids', 'veracitytag', 'text' ]
        train_X = train[train.columns.difference(excluded_features)]
        dev_X = dev[dev.columns.difference(excluded_features)]
        test_X = test[test.columns.difference(excluded_features)]

        train_f = train_X.values.astype(float)
        dev_f = dev_X.values.astype(float)
        test_f = test_X.values.astype(float)


        train_y = dl_data_prep(train_labels)
        test_y = dl_data_prep(test_labels)
        dev_y = dl_data_prep(dev_labels)



        bert_tweet = AutoModel.from_pretrained("vinai/bertweet-base")
        tokenizer_tweet = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
        train_input_ids, train_attention_masks, train_labels = tokenizer_func(tokenizer_tweet, train_sentences,
                                                                              train_labels)
        dev_input_ids, dev_attention_masks, dev_labels = tokenizer_func(tokenizer_tweet, dev_sentences, dev_labels)
        test_input_ids, test_attention_masks, test_labels = tokenizer_func(tokenizer_tweet, test_sentences, test_labels)


        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        val_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

        import time
        t0 = time.time()
        # bert_tweet.cuda()
        bert_tweet.eval()
        # if os.path.isdir('Rumor/train_vect')
        with torch.no_grad():
            if os.path.isfile("data/Rumor/train_vect"):
                print('loading pre generated data')
                train_features = pd.read_pickle('data/Rumor/train_vect')
                train_y = pd.read_pickle('data/Rumor/train_y')
                dev_features = pd.read_pickle('data/Rumor/dev_vect')
                dev_y = pd.read_pickle('data/Rumor/dev_y')
                test_features = pd.read_pickle('data/Rumor/test_vect')
                test_y = pd.read_pickle('data/Rumor/test_y')
            else:
                print('********* \n Generating data from scratch \n *********')
                last_hidden_states_train = bert_tweet(train_input_ids, attention_mask=train_attention_masks)
                train_features = last_hidden_states_train[0][:, 0, :].numpy()
                pd.to_pickle(train_features, 'data/Rumor/train_vect')
                pd.to_pickle(train_y, 'data/Rumor/train_y')
                print('train done')
                last_hidden_states_dev = bert_tweet(dev_input_ids, attention_mask=dev_attention_masks)
                dev_features = last_hidden_states_dev[0][:, 0, :].numpy()
                pd.to_pickle(dev_features, 'data/Rumor/dev_vect')
                pd.to_pickle(dev_y, 'data/Rumor/dev_y')
                print('dev done')
                last_hidden_states_test = bert_tweet(test_input_ids, attention_mask=test_attention_masks)
                test_features = last_hidden_states_test[0][:, 0, :].numpy()
                pd.to_pickle(test_features, 'data/Rumor/test_vect')
                pd.to_pickle(test_y, 'data/Rumor/test_y')
                print('test done')




        source_input = keras.Input(shape=(768,), name="source")
        source_input1 = BatchNormalization()(source_input)
        source_dense2 = layers.Dense(32, activation="tanh")(source_input1)
        # source_output = source_dense(source_dropped_out)
        DO_layer_source = tf.keras.layers.Dropout(0.99, input_shape=(16,))(source_dense2)
        stance_input = keras.Input(shape=(705,),name="stance")
        stance_dense3 = layers.Dense(8, activation="tanh")(stance_input)
        # DO_layer_stance = tf.keras.layers.Dropout(0.1, input_shape=(8,))(stance_dense3)
        concated_input1 = layers.concatenate([stance_dense3, DO_layer_source])
        combined_layer = layers.Dense(16, activation="tanh")(concated_input1)
        label = layers.Dense(3, activation="softmax", name="label")
        verification_prediction = label(combined_layer)
        combined_model = keras.Model(inputs=[source_input, stance_input],outputs=[verification_prediction])

        # combined_model = keras.Model(inputs=[source_input], outputs=[verification_prediction])

        keras.utils.plot_model(combined_model, "multi_input_and_output_model.png", show_shapes=True)
        print('training is strating')
        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint(filepath='old_models/model.{epoch:02d}-{val_loss:.2f}.h5'),
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        ]

        combined_model.compile(
            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer=keras.optimizers.Adagrad(lr=args.lrate),
            metrics=["accuracy"],
        )

        # checkpoint = ModelCheckpoint('model-{epoch:03d}-{accuracy:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
        # history = combined_model.fit({"source":train_features,"stance":stance_train},train_labels_hv, batch_size=16, epochs=500, validation_data=({"source":dev_features,"stance":stance_dev},dev_labels_hv), callbacks=my_callbacks)
        # history = combined_model.fit({"source":train_features,"stance":stance_train},train_labels_hv, batch_size=16, epochs=500, validation_data=({"source":dev_features,"stance":stance_dev},dev_labels_hv), callbacks=my_callbacks)

        history = combined_model.fit({"source":train_features,"stance":train_f}, train_y, batch_size=16, epochs=100, shuffle=True,
                                     validation_data=({"source":dev_features,"stance":dev_f}, dev_y), callbacks=my_callbacks)

        # test_scores = combined_model.evaluate({"source":test_features,"stance":stance_test},test_labels_hv, verbose=2)

        # y_pred_test=combined_model.predict({"source":test_features,"stance":stance_test})
        # y_pred_dev=combined_model.predict({"source":dev_features,"stance":stance_dev})
        y_pred_test = combined_model.predict({"source": test_features, "stance":test_f})
        y_pred_dev = combined_model.predict({"source": dev_features,"stance":dev_f})
        combined_model.summary()
        print("f1 score dev:", f1_h(dev_y, y_pred_dev))
        print("f1 score test:", f1_h(test_y, y_pred_test))
        print('training with transformers')
    elif settging[4] == 'content':
        model_vectorizer = Pipeline(
            [('vect', CountVectorizer(tokenizer=Tokenizer(), ngram_range=(1, 2), max_features= 5000,
                                      stop_words='english')),
             ('tfidf', TfidfTransformer())])
        tf_vect = model_vectorizer.fit_transform(X['text'])
        tf_vect = pd.DataFrame(tf_vect.toarray())
        std = StandardScaler()
        X_tmp = std.fit_transform(tf_vect)
        model = models[arg.model]
        model.fit(X_tmp, list(y_train_upsampled))
        if args.pca:
            model_vectorizer = Pipeline([('vect', CountVectorizer(tokenizer=Tokenizer(), ngram_range=(1, 2), max_features=5000,
                                                       stop_words='english')),
                              ('tfidf', TfidfTransformer())])
            tf_vect=model_vectorizer.fit_transform(X['text'])
            tf_vect=pd.DataFrame(tf_vect.toarray())
            std = StandardScaler()
            X_tmp = std.fit_transform(tf_vect)
            model=Pipeline([('pca',PCA(n_components=292)),('clf',models[arg.model])])
            model.fit(X_tmp, list(y_train_upsampled))

    elif settging[4] == 'all' or settging[4] == 'allnostance':
        model_vectorizer = Pipeline([('vect', CountVectorizer(tokenizer=Tokenizer(), ngram_range=(1, 2), max_features=2000,
                                      stop_words='english')),('tfidf', TfidfTransformer())])
        tf_vect = model_vectorizer.fit_transform(X['text'])
        tf_vect = pd.DataFrame(tf_vect.toarray())
        std = StandardScaler()
        X_tmp = X[X.columns.difference(['text'])]
        tf_vect.reset_index(drop=True, inplace=True)
        X_tmp.reset_index(drop=True, inplace=True)
        train_df = pd.concat([X_tmp, tf_vect], axis=1)
        X_tmp=std.fit_transform(train_df)
        model = models[arg.model]
        if args.pca:
            model = Pipeline([('pca', PCA(n_components=208)), ('clf', models[arg.model])])
        model.fit(X_tmp, list(y_train_upsampled))
        assert len(X.columns) == len(list(set(X.columns))), print("Extra features")

    else:
        std = StandardScaler()
        X_tmp = std.fit_transform(X)
        model=models[arg.model]
        if args.pca:
            pca_p = PCA()
            pca_p.fit_transform(X_tmp)
            numberofcomponenets=len([x for x in pca_p.explained_variance_ if int(x) > 1])
            if numberofcomponenets==0:
                numberofcomponenets=1
            model = Pipeline([('pca', PCA(n_components=numberofcomponenets)), ('clf', models[arg.model])])
        model.fit(X_tmp, list(y_train_upsampled))

    print("saving the model.....")


    filename = 'ver_trained_models/'  +arg.model+ '_'.join([str(x) for x in settging])
    # pickle.dump(model, open(filename, 'wb'))
    print(filename)
    for test_c,test_set in test_set_container.items():
        X_test = test_set[test_set.columns.difference(excluded_features)]
        if settging[4] == 'content':
            tf_vect = model_vectorizer.transform(X_test['text'])
            tf_vect = pd.DataFrame(tf_vect.toarray())
            X_tmp = std.transform(tf_vect)
            y_pred = model.predict(X_tmp)

        elif settging[4] == 'all' or settging[4] == 'allnostance':
            vect_dev_text=model_vectorizer.transform(X_test['text'])
            X_test = X_test[X_test.columns.difference(['text'])]
            text_df=pd.DataFrame(vect_dev_text.toarray())
            text_df.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)
            dev_df = pd.concat([X_test, text_df],axis=1)
            X_tmp = std.transform(dev_df)
            # dev_df.fillna(0, inplace=True)
            y_pred = model.predict(X_tmp)

        else:
            X_tmp = std.transform(X_test)
            # X_test.fillna(0, inplace=True)
            y_pred = model.predict(X_tmp)

        if args.test == 'singletest':
            print(classification_report(list(y_test_set_container['y_'+str(test_c)]), y_pred))

        cReport = classification_report(list(y_test_set_container['y_'+str(test_c)]), y_pred, output_dict=True)

        if task=='ver' and settging[2]=='srcOnly':
            label = [0, 1, 2]
            ymap={0:'False',1:'True',2:'Unverified'}
            res_middle_part=''
            log=logver

        elif task == 'ver' and args.vern:
            label = [0,1,2,3,4,5]
            ymap={0:'False',1:'True',2:'Unverified',3:'SrcTrue',4:'SrcFalse',5:'SrcUnverified'}
            res_middle_part = str(cReport['3']['f1-score']) + ',' + str(cReport['3']['precision']) + ',' + str( \
                cReport['3']['recall']) + ',' + str(cReport['3']['support']) + ',' + \
                              str(cReport['4']['f1-score']) + ',' + str(cReport['4']['precision']) + ',' + str( \
                cReport['4']['recall']) + ',' + str(cReport['4']['support']) + ',' + \
                              str(cReport['5']['f1-score']) + ',' + str(cReport['5']['precision']) + ',' + str( \
                cReport['5']['recall']) + ',' + str(cReport['5']['support']) + ','
            log=logver6
        elif task == 'ver' and not args.vern:
            label = [0,1,2]
            ymap={0:'False',1:'True',2:'Unverified'}
            res_middle_part = ''
            log=logver6n
        else:
            label = [0, 1, 2, 3]
            ymap={0:'Comment',1:'Support',2:'Deny',3:'Query'}
            res_middle_part = str(cReport['3']['f1-score']) + ',' + str(cReport['3']['precision']) + ',' + str( \
                cReport['3']['recall']) + ',' + str(cReport['3']['support']) + ','
            log=logStance
        cm_filename = 'ver_stance_sm_images/' + '_'.join([str(x) for x in settging])+'_'+str(test_c)+"_"+arg.model+'.png'

        key_id = '_'.join([str(x) for x in settging])+'_'+str(test_c)+"_"+arg.model
        cm_analysis(y_test_set_container['y_'+str(test_c)], y_pred,cm_filename,label, ymap=ymap, figsize=(len(label)+2,len(label)+2))
        # cm_test = str(confusion_matrix(y_test, y_test_pred,labels=[0,1,2]))


            # story,feature Category,Time ind,repfeature,frameorcum, Classifier, tn,fn,tp,fp,f1-score 0,precision 0, recall 0, support 0,f1-score 1,precision 1, recall 1, support 1,\
            # f1 macro, precision macro, recall macro, support macro, f1 micro, precision micro, recall micro, support micro,
            # f1 weighted, precision weighted, recall weighted, support weighted
        result_first_part=settging[0] + "," + str(settging[1]) + "," + str(settging[2]) + ',' + str(settging[3]) + ',' + str(settging[4])\
                         + "," + settging[5] + "," + settging[6]+ "," +arg.model + ',' +key_id+','+str(test_c)+','+ \
                      str(cReport['0']['f1-score']) + ',' + str(cReport['0']['precision']) + ',' + str(\
                cReport['0']['recall']) + ',' + str(cReport['0']['support']) + ',' + \
                      str(cReport['1']['f1-score']) + ',' + str(cReport['1']['precision']) + ',' + str(\
                cReport['1']['recall']) + ',' +\
                      str(cReport['1']['support']) + ',' + \
                         str(cReport['2']['f1-score']) + ',' + str(cReport['2']['precision']) + ',' + str(\
                cReport['2']['recall']) + ',' + str(cReport['2']['support']) + ','

        resutl_final_part=str(cReport['macro avg']['f1-score']) + ',' + str(cReport['macro avg']['precision']) + ',' + str(\
                cReport['macro avg']['recall']) + ',' + str(cReport['macro avg']['support']) + ',' + \
                      str(cReport['micro avg']['f1-score']) + ',' + str(cReport['micro avg']['precision']) + ',' + str(\
                cReport['micro avg']['recall']) + ',' + str(cReport['micro avg']['support']) + ',' + \
                      str(cReport['weighted avg']['f1-score']) + ',' + str(\
                cReport['weighted avg']['precision']) + ',' + str(cReport['weighted avg']['recall']) + ',' +\
                      str(cReport['weighted avg']['support']) + '\n'


        all_result_log = result_first_part + res_middle_part + resutl_final_part
        log.write(all_result_log)
        print(all_result_log)


def main(args):
    oldexp={}
    train_pkl="data/Rumor/train_rum_ver_stance_cbed.pkl"
    dev_pkl="data/Rumor/dev_rum_ver_stance_cbed.pkl"
    test_pkl="data/Rumor/test_rum_ver_stance_cbed.pkl"

    df_train=pd.read_pickle(train_pkl)
    df_dev=pd.read_pickle(dev_pkl)
    df_test=pd.read_pickle(test_pkl)
    platforms={'both':2,'twitter':1,'reddit':0} #this is opposite
    tasks=['ver', 'stance']

    ver_task_srcorret=['srcOnly', 'srcNret']

    user = ['follower', 'friend', 'statuses', 'user', 'verified','story','platform','parent_ids','id','src_rtw','veracitytag','parent_body','stance']
    Pragmatic = ['Sentiment', 'Subjecitvity', 'cb_', 'corse','reading_ease','parent_ids', 'na_','id', 'ncb_','platform','src_rtw', 'rob_','parent_body','story','veracitytag','stance']
    twitter = ['Hashtag','hash', 'RetOrRep', 'media', 'sylCount', 'src_rtw','parent_ids','url','id', 'url_ligit','platform','parent_body', 'url_top','story','veracitytag','stance']
    network=['depth','Rate','platform','story','veracitytag','parent_body','parent_ids','stance','id','src_rtw']
    content=['text','veracitytag','platform','src_rtw','story','stance','parent_ids','parent_body','id']
    stance=['src_rtw','sts','stance','story','platform','veracitytag','parent_ids','parent_body','id']


    fram=['frame','cum','ALLFEATURES']
    framdict={'0':'cum', '1':'frame','2':'ALLFEATURES'}
    retresrep = {'ret': ['rep', 'res', 'Res'], 'rep': ['ret', 'RT', 'res', 'Res'], 'res': ['rep', 'ret', 'RT'],
                 'Norep': ['rep', 'res', 'RT', 'ret', 'Res'], 'all': []}
    if args.test=='singletest':
        tasks=args.tasks
        framdict = {args.frame:framdict[args.frame]}
        platforms=args.platform
        ver_task_srcorret=args.srcorRet
        retresrep = args.retresrep
        timeind=args.timeind
        done=True

    TIME_GAP = [0, 0.1, 0.33, 1.2, 72, 168, 700]  # Time gaps for the tests 1 means one hour 0.1 mean 6 minutes
    for platform,opPlatform in platforms.items():
        for task in tasks:
            # if task =='ver':
                for sns in ver_task_srcorret:
                    if sns == 'srcOnly' and task == "stance":
                        continue
                    # sns='srcOnly'
                    for ind, time in enumerate(TIME_GAP):
                        # ind=6
                        # task='ver'
                        if ind != 0:
                            all = user + Pragmatic + network + twitter+content + stance
                            allnostance = user + Pragmatic + network + twitter + content
                            nocontent = user + Pragmatic + network + twitter

                            fcategor = {'user': user, 'Pragmatic': Pragmatic, 'twitter': twitter,'stance':stance, 'content':content,'network': network,
                                        'all': all,'allnostance':allnostance,'nocontent':nocontent}
                        else:
                            all = user + Pragmatic + twitter+content + stance
                            allnostance = user + Pragmatic + twitter + content
                            nocontent = user + Pragmatic + twitter

                            fcategor = {'user': user, 'Pragmatic': Pragmatic, 'twitter': twitter, 'stance': stance, 'content' : content , 'all': all,'allnostance':allnostance,'nocontent':nocontent}

                        if args.test == 'singletest':
                            if not done:
                                break
                            ind=timeind
                            done=False
                            fcategor={str(args.feature):fcategor[args.feature]}

                        print('feature eng is about to begin in ' + task + ' ' + sns + ' and time ' + str(
                            TIME_GAP[ind]) + ' ')

                        if os.path.isfile("data/Rumor/Logistic_regression_vertrain"):
                            if os.path.isfile('data/Rumor/ver_stance_features_'+str(ind)+'train'):
                                train_df = time_frame_feature_generatro('none','train', ind)
                            else:
                                fgen_train_df = featureGen(df_train, 'train', "Nload")
                                train_df = time_frame_feature_generatro(fgen_train_df, 'train', ind)
                        else:
                            fgen_train_df = featureGen(df_train, 'train', "Load")
                            train_df = time_frame_feature_generatro(fgen_train_df, 'train', ind)

                        if os.path.isfile("data/Rumor/Logistic_regression_verdev"):
                            if os.path.isfile('data/Rumor/ver_stance_features_'+str(ind)+'dev'):
                                dev_df = time_frame_feature_generatro('none','dev', ind)
                            else:
                                fgen_dev_df = featureGen(df_dev, 'dev', "Nload")
                                dev_df = time_frame_feature_generatro(fgen_dev_df, 'dev', ind)
                        else:
                            fgen_dev_df = featureGen(df_dev, 'dev', "Load")
                            dev_df = time_frame_feature_generatro(fgen_dev_df, 'dev', ind)

                        if os.path.isfile("data/Rumor/Logistic_regression_vertest"):
                            if os.path.isfile('data/Rumor/ver_stance_features_' + str(ind) + 'test'):
                                test_df = time_frame_feature_generatro('none','test', ind)
                            else:
                                fgen_test_df = featureGen(df_test, 'test', "Nload")
                                test_df = time_frame_feature_generatro(fgen_test_df, 'test', ind)
                        else:
                            fgen_test_df = featureGen(df_test, 'test', "Load")
                            test_df = time_frame_feature_generatro(fgen_test_df, 'test', ind)


                        if sns == 'srcOnly':
                            train_df = train_df[train_df['src_rtw'] == 1]
                            dev_df = dev_df[dev_df['src_rtw'] == 1]
                            test_df = test_df[test_df['src_rtw'] == 1]
                        #limiting to platform
                        print("limiting to platform....")
                        train_df=train_df[train_df['platform']!=opPlatform]
                        # dev_df=dev_df[dev_df['platform']!=opPlatform]
                        # test_df=test_df[test_df['platform']!=opPlatform]


                        print('dropping the columns with only one value in it and list of those columns are...')
                        singular_value_columns=[x for x in train_df.columns if len(train_df[x].unique()) == 1]
                        if 'platform' in singular_value_columns:
                            #we need this later for filtering the dev and test set
                            singular_value_columns.remove('platform')
                        train_df.drop(columns=singular_value_columns,inplace=True)
                        # print(singular_value_columns)

                        columns = train_df.columns
                        logresultver = open('results/ver_log_pca_false' + args.model + '.txt', 'a+')
                        logresultstance = open('results/stance_log_pca_false' + args.model + '.txt', 'a+')
                        logresultver6 = open('results/ver6_log_pca_false' + args.model + '.txt', 'a+')
                        logresultver6n = open('results/ver6n_log_pca_false' + args.model + '.txt', 'a+')

                        for k, v in fcategor.items():
                            # k='all'
                            # v=fcategor[k]
                            if ind != 0:
                                for tw_typ, values in retresrep.items():
                                    for indf, framecategory in enumerate(fram):
                                        # framecategory='cum'
                                        print(k+' '+framecategory+' '+tw_typ+' '+sns+' '+task+' '+platform)
                                        v = list(set(v))
                                        selected_features = {}
                                        if args.test=='singletest':#Just for the test mode to overwrite the index for frame selection
                                            indf=int(args.frame)
                                        for f_s in v:
                                            if k+str(ind) in selected_features.keys():
                                                scol=[x for x in columns if f_s in x and framdict[str(indf)] not in x ]
                                                replied_feature=[x for y in values for x in scol if y in x]
                                                selected_features[k+str(ind)].extend([x for x in scol if x not in replied_feature])
                                            else:
                                                scol=[x for x in columns if f_s in x and framdict[str(indf)] not in x ]
                                                replied_feature=[x for y in values for x in scol if y in x]
                                                selected_features[k+str(ind)]=[x for x in scol if x not in replied_feature]
                                        traindata = train_df[selected_features[k+str(ind)]]
                                        devdata = dev_df[selected_features[k+str(ind)]]
                                        testdata = test_df[selected_features[k+str(ind)]]
                                        exp_set=[platform,task,sns,ind,k,tw_typ,framecategory]
                                        exp_kID="".join([str(x) for x in exp_set])
                                        if exp_kID in oldexp.keys():
                                            continue
                                        else:
                                            oldexp[exp_kID]=1
                                        train_test(traindata, devdata, testdata, args, exp_set, logresultver,
                                                       logresultstance, logresultver6, logresultver6n)
                                        # # print("List of the selected features for this task are: ")
                                        # print(selected_features[k+str(ind)])
                            else:
                                selected_features = {}
                                for f_s in v:
                                    if k + str(ind) in selected_features.keys():
                                        scol = [x for x in columns if f_s in x ]
                                        selected_features[k + str(ind)].extend(scol)
                                    else:
                                        scol = [x for x in columns if f_s in x ]
                                        selected_features[k + str(ind)] = scol
                                traindata = train_df[selected_features[k + str(ind)]]
                                devdata = dev_df[selected_features[k + str(ind)]]
                                testdata = test_df[selected_features[k + str(ind)]]

                                exp_set = [platform, task, sns, ind, k, 'noresrep', 'Noframe']
                                exp_kID = "".join([str(x) for x in exp_set])
                                if exp_kID in oldexp.keys():
                                    continue
                                else:
                                    oldexp[exp_kID] = 1
                                train_test(traindata, devdata, testdata, args, exp_set, logresultver, logresultstance,logresultver6,logresultver6n)
                                # 
                                #
                                # print("List of the selected features for this task are: ")
                                # print(selected_features[k + str(ind)])

                        logresultver.close()
                        logresultstance.close()
                        logresultver6.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='transformer',help='model for classification')
    parser.add_argument('--frame',  default='2',help="select if you want the feature frame ['frame' 0,'cum'1 ,'ALLFEATURES'2]")
    parser.add_argument('--tasks', default=['ver'],help="select the task eaither stance or Veracity ['stance','ver']'")
    parser.add_argument('--retresrep', default={'all': []},help='res or rep or ret or all')    # 'ret': ['rep', 'res', 'Res'], 'rep': ['ret', 'RT', 'res', 'Res'], 'res': ['rep', 'ret', 'RT'],
    # 'Norep': ['rep', 'res', 'RT', 'ret', 'Res'],
    parser.add_argument('--srcorRet',default=['srcNret'],help="source or with retweets['srcOnly','srcNret'] ")
    parser.add_argument('--vern',default=False,help="This is to conver 6 way classification to three also using replies")
    parser.add_argument('--timeind', default=6,help='what time index you want ti get the feature of')
    parser.add_argument('--feature', type=str, default='all',help='the category of feature e.g., user, Pragmatic, twitter,network,content ..')
    parser.add_argument('--testSet', default=None,help='if you want to test on certain test set')
    parser.add_argument('--platform', default= {'both':2} )#this is opposite,'twitter':1,'reddit':0} help='if you want to test on certain test set')
    parser.add_argument('--test', type=str,default='singletest',help='if you want to test on certain test set   singletest')
    parser.add_argument('--dldata',default=True,help='if you want to creat the dataframe with specified features in default for the deep learning pipeline')
    parser.add_argument('--pca',default=True,help='if you want to creat the dataframe with specified features in default for the deep learning pipeline')
    parser.add_argument('--lrate',default=0.001,help='learning rage for the dlearning model')


    # 'ret': ['rep', 'res', 'Res'], 'rep': ['ret', 'RT', 'res', 'Res'], 'res': ['rep', 'ret', 'RT'],
    # 'Norep': ['rep', 'res', 'RT', 'ret', 'Res'],

    args = parser.parse_args()
    task='ver'
    train_set_path='mtl_feature/train.txt'
    dev_set_path='mtl_feature/dev.txt'
    test_set_path='mtl_feature/test.txt'
    main(args)




        # nX = X[X.columns.difference(['id', 'story','rum_tag','length','created','reading_ease_parent','RetOrRep_parent','parent_ids',\
        #                          'Sentiment_parent','Subjecitvity_parent','subnode_max_depth','depth','timeGap',\
        #                          'src_rtw','ups',])]





