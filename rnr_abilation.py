"""This method code is to run multiple machine learning experiments and find the best fit model for the task of rumor identification"""
import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn import svm
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix


def upsampler(datframe):
    df_majority = datframe[datframe.rum_tag == 1]  # Rum
    df_minority = datframe[datframe.rum_tag == 0]
    maj_len=df_majority.shape[0]
    min_len=df_minority.shape[0]
    size=maj_len
    minor=df_minority
    major=df_majority
    if maj_len<min_len:
        size=min_len
        minor=df_majority
        major=df_minority
    # NR
    minor = resample(minor,replace=True,  # sample with replacement
                                     n_samples=size,  # to match majority class
                                     random_state=123)
    df_upsampled = pd.concat([major, minor])
    return df_upsampled

pattern = r'\w+|\?|\!|\"|\'|\;|\:'
class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(pattern)
        self.stemmer = stemmer
    def __call__(self, doc):
        return [self.stemmer.stem(token)
                for token in self.tok.tokenize(doc)]

stemmer = SnowballStemmer("english")


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
    plt.savefig(filename+".pdf")
    plt.show()
    plt.close()

def cross_validation(model,data,model_type,features,frameorcum,timeind,repfeature,log):
    """Performs the cross validation only for rnr by taking one story out train and test on the rest
    """
    # drop_stories=['gurlitt-all-rnr-threads', 'putinmissing-all-rnr-threads', 'prince-toronto-all-rnr-threads', 'ebola-essien-all-rnr-threads']
    stories=['charliehebdo-all-rnr-threads', 'ottawashooting-all-rnr-threads',
       'ferguson-all-rnr-threads', 'germanwings-crash-all-rnr-threads','sydneysiege-all-rnr-threads']
    scors={}
    y_gold=[]
    y_all_pred=[]
    for item in stories:
        train_set = data[(data['story'] != 'gurlitt-all-rnr-threads') & (data['story'] != 'putinmissing-all-rnr-threads')& \
                       (data['story'] != 'prince-toronto-all-rnr-threads')& (data['story'] != 'ebola-essien-all-rnr-threads')\
            & (data['story'] != item)]
        train_set = upsampler(train_set)

        test = data[data['story'] == item]
        y = train_set['rum_tag']
        yt = test['rum_tag']
        excluded_features=['id', 'story', 'num_comment', 'rum_tag', 'veracitytag', 'length', 'created', 'reading_ease_parent',
             'RetOrRep_parent', 'parent_ids', \
             'Sentiment_parent', 'Subjecitvity_parent', 'subnode_max_depth', 'depth', 'timeGap', \
             'src_rtw', 'ups']

        X = train_set[train_set.columns.difference(excluded_features)]
        Xt = test[test.columns.difference(excluded_features)]
        if features == 'content':
            model_vectorizer = Pipeline(
                [('vect', CountVectorizer(tokenizer=Tokenizer(), ngram_range=(1, 2), max_features=2000,
                                          stop_words='english')),
                 ('tfidf', TfidfTransformer())])
            tf_vect = model_vectorizer.fit_transform(X['text'])
            tf_vect = pd.DataFrame(tf_vect.toarray())
            std = StandardScaler()
            X_tmp = std.fit_transform(tf_vect)
            model = Pipeline([('pca', PCA()), ('clf', model)])
            model.fit(X_tmp, list(y))
            #Fro Test
            test_vect=model_vectorizer.transform(Xt['text'])
            test_vect_df = pd.DataFrame(test_vect.toarray())
            Xt=std.transform(test_vect_df)
        elif features == 'all':
            model_vectorizer = Pipeline(
                [('vect', CountVectorizer(tokenizer=Tokenizer(), ngram_range=(1, 2), max_features=2000,
                                          stop_words='english')), ('tfidf', TfidfTransformer())])
            tf_vect = model_vectorizer.fit_transform(X['text'])
            tf_vect = pd.DataFrame(tf_vect.toarray())
            std = StandardScaler()
            X_tmp = X[X.columns.difference(['text'])]
            tf_vect.reset_index(drop=True, inplace=True)
            X_tmp.reset_index(drop=True, inplace=True)
            train_df = pd.concat([X_tmp, tf_vect], axis=1)
            X_tmp = std.fit_transform(train_df)
            model = Pipeline([('pca', PCA()), ('clf', model)])
            model.fit(X_tmp, list(y))
            #fortest
            Xt_temp=Xt[Xt.columns.difference(['text'])]
            Xt_text=model_vectorizer.transform(Xt['text'])
            # X_test = Xt_text[Xt_text.columns.difference(['text'])]
            text_df=pd.DataFrame(Xt_text.toarray())
            text_df.reset_index(drop=True, inplace=True)
            Xt_temp.reset_index(drop=True, inplace=True)
            dev_df = pd.concat([Xt_temp, text_df],axis=1)
            Xt = std.transform(dev_df)
            # dev_df.fillna(0, inplace=True)

        else:
            print('training is going to start...')
            std = StandardScaler()
            X=std.fit_transform(X)
            model.fit(X,y)
            Xt=std.transform(Xt)
            print('training is done')

        y_pred = model.predict(Xt)
        y_gold.extend(yt)
        y_all_pred.extend(y_pred)
        cm=confusion_matrix(yt, y_pred)
        tn = str(cm[0][0])
        fn = str(cm[1][0])
        tp = str(cm[1][1])
        fp = str(cm[0][1])
        cReport = classification_report(yt, y_pred, output_dict=True)
        #story,feature Category,Time ind,repfeature,frameorcum, Classifier, tn,fn,tp,fp,f1-score 0,precision 0, recall 0, support 0,f1-score 1,precision 1, recall 1, support 1,\
        #f1 macro, precision macro, recall macro, support macro, f1 micro, precision micro, recall micro, support micro,
        # f1 weighted, precision weighted, recall weighted, support weighted

        log.write(item+","+str(features)+","+str(timeind)+','+str(repfeature)+','+str(frameorcum)+","+model_type +","+tn+','+fn+','+tp+','+fp+','+ \
                  str(cReport['0']['f1-score'])+','+str(cReport['0']['precision'])+','+str(cReport['0']['recall'])+','+str(cReport['0']['support'])+','+ \
                  str(cReport['1']['f1-score']) + ',' + str(cReport['1']['precision']) + ',' + str(cReport['1']['recall']) + ',' +
                  str(cReport['1']['support']) + ','+\
                  str(cReport['macro avg']['f1-score']) + ',' + str(cReport['macro avg']['precision']) + ',' + str(
                    cReport['macro avg']['recall']) + ',' + str(cReport['macro avg']['support'])+ ','+\
                  str(cReport['micro avg']['f1-score']) + ',' + str(cReport['micro avg']['precision']) + ',' + str(
                    cReport['micro avg']['recall']) + ',' +str(cReport['micro avg']['support']) +','+ \
                  str(cReport['weighted avg']['f1-score']) + ',' + str(
                    cReport['weighted avg']['precision']) + ',' + str(cReport['weighted avg']['recall']) + ',' +
                  str(cReport['weighted avg']['support']) + '\n')
    cm=confusion_matrix(y_gold, y_all_pred)
    tn = str(cm[0][0])
    fn = str(cm[1][0])
    tp = str(cm[1][1])
    fp = str(cm[0][1])
    cReport = classification_report(y_gold, y_all_pred, output_dict=True)

    log.write("All stories" + "," + str(features) + "," + str(
        timeind) + ','+str(repfeature)+','+str(frameorcum)+","+ model_type + "," + tn + ',' + fn + ',' + tp + ',' + fp + ',' + \
              str(cReport['0']['f1-score']) + ',' + str(cReport['0']['precision']) + ',' + str(
        cReport['0']['recall']) + ',' + str(cReport['0']['support']) + ','+\
              str(cReport['1']['f1-score']) + ',' + str(cReport['1']['precision']) + ',' + str(
        cReport['1']['recall']) + ',' +
              str(cReport['1']['support']) +','+ \
              str(cReport['macro avg']['f1-score']) + ',' + str(cReport['macro avg']['precision']) + ',' + str(
        cReport['macro avg']['recall']) + ',' + str(cReport['macro avg']['support']) +','+ \
              str(cReport['micro avg']['f1-score']) + ',' + str(cReport['micro avg']['precision']) + ',' + str(
        cReport['micro avg']['recall']) + ',' + str(cReport['micro avg']['support']) +','+ \
              str(cReport['weighted avg']['f1-score']) + ',' + str(
        cReport['weighted avg']['precision']) + ',' + str(cReport['weighted avg']['recall']) + ',' +
              str(cReport['weighted avg']['support']) + '\n')

    # return scors

    ymap = {0: 'Non-Rumor', 1: 'Rumor'}
    label=[0,1]
    cm_analysis(y_gold, y_all_pred, "images/"+args.name, label, ymap=ymap,figsize=(4, 4))


def load_pickle(path):
    try:
        return pd.read_pickle(path)
    except:
        print("Error reading the pickle file")

def main(args):
    if TEST == 'rnr':
        TIME_GAP_bench = [0,0.1,0.44,2,72,168,900]  # Time gaps for the tests 1 means one hour 0.1 mean 6 minutes
        TIME_GAP=[0,0.1,0.44,2,700]

    elif TEST == 'ver':
        TIME_GAP_bench = [0, 72, 164, 280, 672]
        TIME_GAP = [0, 0.1, 0.33, 1.2, 900]  # Time gaps for the tests 1 means one hour 0.1 mean 6 minutes

    for ind,time in enumerate(TIME_GAP_bench):
        if args.test == 'singletest':
            ind = args.timeind
        if TEST=='rnr':
            nX=pd.read_pickle('data/Rumor/Logistic_full_features_'+TEST+str(ind))
        else:
            nX = pd.read_pickle('data/Rumor/Logistic_full_features_day_' + TEST + str(ind))

        # X=X.sample(frac=0.001, replace=True, random_state=1)#Just added and should be commented out
        nX = nX[nX['src_rtw'] == 'src']
        nX['rum_tag'] = np.where(nX['rum_tag'] == 1, 0, 1)
        print("singluar matrix are....")
        print([str(x) + " : " + str(len(nX[x].unique())) for x in list(nX.columns)])
        nX.fillna(nX.mean(), inplace=True)

        X_train=nX
        columns = X_train.columns

        os_data_X = pd.DataFrame(data=nX, columns=columns)
        os_data_X.drop(columns=[x for x in os_data_X.columns if len(os_data_X[x].unique()) == 1], inplace=True)

        models={'svm':svm.SVC(gamma='auto'),'Rf':RandomForestClassifier(n_estimators=100, max_depth=20),\
                'NB':GaussianNB(),'MLP': MLPClassifier(alpha=1),'ADA':AdaBoostClassifier()}
        retresrep={'ret':['_rep','_res','_Res'],'rep':['_ret','_RT','_res','_Res'],'res':['_rep','_ret','_RT'],'Norep':['_rep', '_res', '_RT', '_ret', '_Res'],'all':[]}
        user = ['follower', 'friend', 'statuses', 'user', 'verified','story','rum_tag']
        Pragmatic = ['Sentiment', 'Subjecitvity', 'cb_', 'corse', 'na_', 'ncb_', 'reading_ease', 'rob_','story','rum_tag']
        twitter = ['Hashtag','hash', 'RetOrRep', 'media', 'sylCount', 'url', 'url_ligit', 'url_top','story','rum_tag']
        network=['depth','Rate','story','rum_tag']
        content = ['text', 'story','rum_tag' ]

        fram=['frame','cum','ALLFEATURES']
        framdict={'0':'cum', '1':'frame','2':'ALLFEATURES'}
        columns = os_data_X.columns

        if ind != 0:
            all = user + Pragmatic + network + twitter + content
            nocontent = user + Pragmatic + network + twitter
            
            fcategor = {'user': user, 'Pragmatic': Pragmatic, 'twitter': twitter,
                        'network': network,
                        'all': all, 'nocontent': nocontent}
        else:
            all = user + Pragmatic + twitter + content
            nocontent = user + Pragmatic + twitter

            fcategor = {'user': user, 'Pragmatic': Pragmatic, 'twitter': twitter,
                        'all': all, 'nocontent': nocontent,'content':content}
            # all = user + Pragmatic + twitter + content
            # fcategor = {'all': all}
        if args.test == 'singletest':
            fcategor = {str(args.feature): fcategor[args.feature]}
            retresrep = args.retresrep
            done = True



        print("New time index is about to begin")
        for tw_typ,values in retresrep.items():
            for indf,framecategory in enumerate(fram):
                if args.test == 'singletest':
                    if not done:
                        break
                    done = False
                    fcategor = {str(args.feature): fcategor[args.feature]}
                for k,v in fcategor.items():
                    v=list(set(v))
                    selected_features = {}
                    result = open('results/rnr_log_round2_'+args.model+'.txt', 'a+')
                    if args.test == 'singletest':  # Just for the test mode to overwrite the index for frame selection
                        indf = int(args.frame)

                    # result.write("Feature category is: "+ k +" the time ind is "+ str(ind))
                    for f_s in v:
                        if k+str(ind) in selected_features.keys():
                            scol=[x for x in columns if f_s in x and framdict[str(indf)] not in x ]
                            replied_feature=[x for y in values for x in scol if y in x]
                            selected_features[k+str(ind)].extend([x for x in scol if x not in replied_feature])
                        else:
                            scol=[x for x in columns if f_s in x and framdict[str(indf)] not in x ]
                            replied_feature=[x for y in values for x in scol if y in x]
                            selected_features[k+str(ind)]=[x for x in scol if x not in replied_feature]
                    data = os_data_X[selected_features[k+str(ind)]]
                    print("List of the selected features for this task are: ")
                    print(selected_features[k+str(ind)])



                    clf = models[args.model]

                    cross_validation(clf, data,args.model,k,ind,tw_typ,framecategory,result)


                    print('first feature category is done')

                    result.close()
                # exit()#this for content

        # print("fitting the model with logistic regression .....")
        # logit_model = sm.Logit(os_data_y, data)
        # result = logit_model.fit()
        # print(result.summary2())




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='svm',help='model for classification')
    parser.add_argument('--frame',  default='0',help="select if you want the feature frame ['frame' 0,'cum'1 ,'ALLFEATURES'2]")
    parser.add_argument('--retresrep', default={'ret': ['rep', 'res', 'Res']},help='res or rep or ret or all')    # 'ret': ['rep', 'res', 'Res'], 'rep': ['ret', 'RT', 'res', 'Res'], 'res': ['rep', 'ret', 'RT'],
    # 'Norep': ['rep', 'res', 'RT', 'ret', 'Res'],
    parser.add_argument('--srcorRet',default=['srcNret'],help="source or with retweets['srcOnly','srcNret'] ")
    parser.add_argument('--vern',default=False,help="This is to conver 6 way classification to three also using replies")
    parser.add_argument('--timeind', default=0,help='what time index you want ti get the feature of')
    parser.add_argument('--feature', type=str, default='user',help='the category of feature e.g., user, Pragmatic, twitter,network,content ..')
    parser.add_argument('--testSet', default=None,help='if you want to test on certain test set')
    parser.add_argument('--test', type=str,default='singletest',help='if you want to test on certain test set   singletest')
    parser.add_argument('--name', type=str,default='user',help='if you want to test on certain test set   singletest')


    args = parser.parse_args()
    # arg_path=args.args
    TEST = 'rnr'
    main(args)


