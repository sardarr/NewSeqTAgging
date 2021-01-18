"""
This file containes all the mthods used for feature extraction of rumour pipeline
Sardar Hamidian
02/11/2019
"""
from __future__ import division, print_function
import json
import pickle
import re
from stanfordcorenlp import StanfordCoreNLP
from curses.ascii import isprint
import numpy as np
import os
from scipy import spatial


GLOV_PATH='FACTMATA/CrossNet/glove/glove.twitter.27B.200d.txt'
EMBEDDING_SIZE = 200
VOCAB_SIZE = 5000
GLOVE_VECS = os.path.join(GLOV_PATH)


# read glove vectors
# glove = collections.defaultdict(lambda: np.zeros((EMBEDDING_SIZE,)))
# fglove = open(GLOVE_VECS, "rb")
# print("loading the embedding....")
# for line in fglove:
#     cols = line.strip().split()
#     word = cols[0].decode('utf-8')
#     embedding = np.array(cols[1:], dtype="float32")
#     glove[word] = embedding
# fglove.close()
# print("loading the embedding is don")

def load_glove_matrix(vec_file):
    if os.path.isfile('data/Rumor/glov.npy'):
        word2vec=np.load('data/Rumor/glov.npy', allow_pickle=True)[()]
        print('Found %s word vectors.' % len(word2vec))
    else:
        word2vec = {}
        with open(vec_file, encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word2vec[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(word2vec))
        np.save('data/Rumor/glov.npy',word2vec)
    return word2vec

glove=load_glove_matrix(GLOVE_VECS)

def corNlpconnect():
    host = 'http://localhost'
    port = 9000
    nlp = StanfordCoreNLP(host, port=port, timeout=30000)
    return nlp

def subtree_finder(d,key,stree=None):
    """
    :param d: The main json
    :param tree: this tree contains the keys and depth of the node in tree
    :param level: the level of the current d starting from
    :return: a dictionary which contains keys and level in the tree for all the node in the tree
    """
    if stree==None:
        stree={}
    if isinstance(d,dict):
        for k,v in d.items():
            if k==key and isinstance(v,dict):
                stree[key]=v
            elif k==key:
                stree[key] = {}
            elif k!=key and isinstance(v,dict):
                subtree_finder(v,key,stree=stree)
    return stree

def clean_tweet(tweet):
    tweet = re.sub('http\S+\s*', '', tweet)  # remove URLs
    tweet = re.sub('RT|cc', '. ', tweet)  # remove RT and cc
    tweet = re.sub('#\S+', '', tweet)  # remove hashtags
    tweet = re.sub('@\S+', '', tweet)  # remove mentions
    tweet = re.sub('[%s]' % re.escape(""""#%&'()*+,-./:;<=>@[\]^_`{|}~"""), '', tweet)  # remove punctuations
    tweet = re.sub('\s+', ' ', tweet)  # remove extra whitespace
    return tweet

def RetOrRep(tweet):
    """
    Args:
        tweet: Raw tweet
    Returns: Identifies if it is tweet retweet or ressponse
    """
    if "RT " in tweet:
        return 2
    elif " @" in tweet:
        return 1
    return 0

def sts(s1,s2):
    """
    Args:
        s1: Gets two sentence and finds the similarity between two
        s2: sentence 2
    Returns:
    """
    def doctovec(splitted):
        num_words = len(splitted)
        if num_words == 0:
            return
        embeddings = np.zeros((len(splitted), EMBEDDING_SIZE))
        for ind, word in enumerate(splitted):
            if word.lower() in glove.keys():
                embeddings[ind, :] = glove[word.lower()]
            else:
                num_words-=1
        return np.sum(embeddings, axis=0) / num_words+1

    text1 = s1.split(" ")
    text2=s2.split(" ")
    vec1 = doctovec(text1)
    vec2=doctovec(text2)
    try:
        sim=1 - spatial.distance.cosine(vec1, vec2)
        return sim
    except:
        return None

def dldatagen(data,path):
    """
    Args:
        data: The dataframe as input
        path: The path to store the dataset

    Returns:

    """
    for ind,row in data.iterrows():
        text=row['text']



def mytree(d,tree=None,level=0,id=""):
    """
    :param d: The main json
    :param tree: this tree contains the keys and depth of the node in tree
    :param level: the level of the curren d starting fromm
    :return: a dictionary which contains keys and level in the tree
    """
    if tree==None:
        tree={}
        tree[id+'_max']=0
    if tree[id+'_max']<level and isinstance(d,dict):
        tree[id+'_max']=level
    if isinstance(d,dict):
        for k, v in d.items():
            tree[k] = level
            mytree(v,tree,level+1,id)
    elif isinstance(d,list):
        for k in d:
            tree[k] = level
            mytree(k, tree, level + 1,id)
    return tree,max



def prep(text):
    """ Same preprocessing used for the belief"""
    TAG_RE = re.compile(r'<[^>]+>')  # removing xml tags
  # removing urls
    first_clean = TAG_RE.sub('', text)
    TAG_RE_URL = re.compile(
        r'\"?h*H*ttp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\"?')
    urls_free = TAG_RE_URL.sub(' url ', first_clean)
    datefre = re.compile(r'\d*/\d*/\d*')
    dtfree = datefre.sub(' ', urls_free)
    numfre = re.compile(r'\-*\,*\d{1,4}\.*\,*\-*\d{1,4}')
    numbfree = numfre.sub(' ', dtfree)
    puncmid = re.compile(r'\ *\w{0}?[^\w\s\?\.]{2,}\ *')
    multpunc = puncmid.sub(' ', numbfree)
    apEnd = re.compile(r'\w{0}?[^\w\s\?\.]{2,}\n')
    dotretclean = apEnd.sub('', multpunc)
    spcefree = re.compile(r'\s{2,}')
    dotclean = spcefree.sub(' ', dotretclean)
    dashf = re.compile(r'\-+')
    preped = dashf.sub(' ', dotclean)
    # Now it should be tried to see if changes have been made correctly
    return preped

def word_divider(tweet):
    return re.sub(r"[_@#\-&$]+\ *", " ", tweet)

# def url_finder(token):
#     if len(token)>10:

def asci_checker(text):
    text=''.join(char for char in text if isprint(char))
    return ''.join([i if ord(i) < 128 else '' for i in text])

def fct_check(word):
    fact_checking = ('snopes', 'politifact', 'factcheck')
    for item in fact_checking:
        if item in word.lower():
            return item
    else:
        return word
def urlChecker(path_top,path_ligit):
    """
    Args:
        path_top: Path to the file which contains the top 500 populat domain by https://moz.com/top500
        path_ligit: This contains the url fact labled by http://www.opensources.co/

    Returns:

    """
    all_labels={'bias': 4,'blog': 1,'clickbait': 12,'conspiracy': 5,'fake': 16,
     'hate': 14,'junksci': 2,'political': 6,'reliable': 8,'rumor': 9,'satire': 15,
     'state': 13,'unreliable': 11}
    lines=open(path_top).readlines()
    domain_table={'':50}
    domain_ligit_table={'':60}
    for domain in lines[1:]:
        splited=domain.split(",")
        domin=splited[1].split('.')[0].replace('"', '')
        domain_table[domin]=int(splited[0])
    lines_ = open(path_ligit).readlines()
    for domain in lines_[1:]:
        splited=domain.split(",")
        domin=splited[0].split('.')[0].replace('"', '')
        tag=''
        if splited[1] in all_labels:
            tag+=str(all_labels[splited[1]])
        elif splited[2] in all_labels:
            tag+=str(all_labels[splited[2]])
        elif splited[3] in all_labels:
            tag+=str(all_labels[splited[3]])
        domain_ligit_table[domin]=int(tag)
    return domain_table,domain_ligit_table

def top_url_returner(key,table):
    """
    This method gets table of top 500 urls and returns the index of the url
    Args:
        key: 'cnn'
        table: 44

    Returns:

    """
    for domains in key:
        if domains in table.keys():
            return table[domains]
    return 501

def ligit_url_returner(key,table):
    """
    This table gets the table of urls and retunrs the binary tag associated with legitimacy of the domain
    Args:
        key:'cnn505'
        table: 14 which is hate for example

    Returns:

    """
    for domains in key:
        if domains in table.keys():
            return table[domains]
    return 17


def uppercase_divider(sentece):
    tokens=[]
    for items in sentece.strip().split(" "):
        if items!=" ":
            up_count=sum(1 for c in items if c.isupper())
            if up_count==len(items):
                tokens.append(items.lower())
            elif up_count<len(items) and up_count>1:
                word=''
                for ch in items:
                    if ch.isupper():
                        word+=' '+ch
                    else:
                        word+=ch
                if word[0]==' ':
                    tokens.append(word[1:])
                else:
                    tokens.append(word)
            else:
                tokens.append(items)
    return " ".join(tokens)

def stanfordHandler(sentece):
    nlp=corNlpconnect()
    postag=[p[1] for p in nlp.pos_tag(sentece)]
    res = nlp.annotate(sentece,properties={'annotators': 'sentiment','outputFormat': 'json','timeout': 10000,})
    sentDist=json.loads(res)["sentences"][0]['sentimentDistribution']
    sentag=json.loads(res)["sentences"][0]['sentiment']
    return [postag,sentDist,sentag]

def entityExtractor(content):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', content, flags=re.MULTILINE)

    urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', content)
    TAG_RE_URL = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
    url_free = TAG_RE_URL.sub(' url ', text)
    return [urls,url_free]

def indexOf(char,string):
    """
    Args:
        char: The char that you are looking for i.e. ',' or '.'
        string: Website
    Returns: [indeces of zeros]

    """
    return [i for i,c in enumerate(string) if c==char]

def domain_finder(url,udil):
    """
    Args:
        url: www.google.com\extension.asd..
        udil: [8,17,20,23,24]
    Returns: the domain between the first two dots
    """
    if len(url)<3:
        return ''
    if len(udil)>1:
        return url[udil[0]+1:udil[1]]
    elif(len(udil))==1:
        start=indexOf('/',url[:10])[-1]+1
        return url[start:udil[0]]
    else:
        return url


def url_Fettcher(urls):
    url_list = []
    for k, v in urls.items():
        if isinstance(urls[k], list):
            [url_list.append(x['expanded_url']) for x in urls[k]]
    return url_list


def url_extraction(json,domain="tw"):
    """
    Args:
        content: The content with url
        meta: The meta inforamtion of the twitter or other platforms
    Returns: list of the [URLs,Inside/Outside URL, News Source URL, Factchecking URL, URL Verify, Category of URL]

    """
    url_list = []
    def url_Fettcher(urls):
        for url in urls:
            if isinstance(url,dict):
                url_list.append(url['expanded_url'])
    if domain=="tw":
        urls=json['entities']['urls']
        hash_tags=json['entities']['hashtags']
        url_Fettcher(urls)
        domains=[domain_finder(url,indexOf('.', url)) for url in url_list]
    elif domain=='reddit':
        if 'children' in json['data'].keys():
            urls = json['data']['children'][0]['data']['url']
            if isinstance(urls, str):
                domains=[domain_finder(urls, indexOf('.', urls))]
            else:
                domains=[domain_finder(url, indexOf('.', url)) for url in urls]
        else:
            text=json['data']['body']
            content_urls=entityExtractor(text)
            domains=[domain_finder(url, indexOf('.', url)) for url in content_urls[0]]
    return domains

def hashtags_extraction(json):
    """
    Args:
        content: The content with url
        meta: The meta inforamtion of the twitter or other platforms
    Returns: list of the [URLs,Inside/Outside URL, News Source URL, Factchecking URL, URL Verify, Category of URL]

    """
    if 'children' in json['data'].keys():
        urls = json['data']['children'][0]['data']['url']
        if isinstance(urls, str):
            domains=[domain_finder(urls, indexOf('.', urls))]
        else:
            domains=[domain_finder(url, indexOf('.', url)) for url in urls]
    else:
        text=json['data']['body']
        content_urls=entityExtractor(text)
        domains=[domain_finder(url, indexOf('.', url)) for url in content_urls[0]]
    return domains




