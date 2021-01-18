"""
Sardar Hamidian
This code is to generate a dataframe with rumors and non-rumors tweet and other attributes collected from meta information of the tweets
"""
import os
import json
import time

import botometer
import pandas as pd
from Feature_extractor import mytree, subtree_finder, stanfordHandler, entityExtractor, prep, word_divider, fct_check, \
    url_extraction

def rumtagFinder(path):
    if 'non-rumours' in path:
        return 'NRum'
    else:
        return 'Rum'

def actionReactionFinder(path):
    if 'reactions' in path:
        return 'react'
    else:
        return 'srcTt'
def depth(d, level=0):
    "Having the dictionary returns the depth"
    if not isinstance(d, dict) or not d:
        return level
    return max(depth(d[k], level + 1) for k in d)

list_of_stories=['charliehebdo-all-rnr-threads','ottawashooting-all-rnr-threads',
'ebola-essien-all-rnr-threads','prince-toronto-all-rnr-threads','ferguson-all-rnr-threads','putinmissing-all-rnr-threads',\
                 'germanwings-crash-all-rnr-threads','sydneysiege-all-rnr-threads','gurlitt-all-rnr-threads']
progCounter=0
# def main():
parent_text={}
parents_full_json={}
all_depth_and_max={}
ver_rnr_tag={}#This is the veracity tag dic which will be filled by annotation.json file
if __name__ == '__main__':
    list=[]
    old={}
    annotation=""

    columns = ['fname', 'title', 'text', 'platform', 'authour', 'id', 'created', 'ups', 'upvote_ratio', 'num_comment',
               'depth', 'edite', 'score', 'url', 'user_json', 'verified', \
               'rum_tag', 'veracitytag', 'parent_body', 'stance', 'ancestor_body', 'kids_body', 'parent_ids',
               'kids_ids', 'sibilings_ids', 'src_rtw', 'previous_tweet_body', 'first_tweet_body', 'subtaskaenglish',
               'subtaskbenglish','story', \
               'burst_time', 'tweet_prop_rate', 'dv_prog_ret', 'dv_prog_rep', 'dev_sup', 'dv_deny', 'dv_ques',
               'df_comment', 'dv_neg', 'picture_url', \
               'sentiment', 'sent_dist', 'emotion', 'Belief', 'Hashtag', 'tree_max_depth', 'Hashtag_ex', 'url_dsrc',
               'Hate_speech', 'Negation', 'polite_spch', 'certainty', 'emotion', 'all_tweets', 'all_urls', 'all_pics',
               'media', 'pic_url', \
               'fcheck_web', 'action_verb', 'pic_caption', 'length', 'pos', 'parser', 'followers',
               'ave_fiends_followes', 'followers_av_followers', 'ave_retweet', 'av_firends_ret', \
               'bot', 'highest_num_ret', 'lowest_num_ret', 'favourites_count', 'usr_location', 'tweet_location',
               'timeGap', 'subnode_max_depth', 'src_tweet_location', \
               'user_time_zone', 'user_created_at', 'user_description', 'listed_count', 'statuses_count',
               'friends_count', 'profile_image_url', 'created_at', 'user_url']

    # columns = ['text', 'id', 'in_reply_to_status_id','entities','user','created_at','lang','source','rumorTag','srcOrReact','story','cb','ev','cert']
    txtdf=['text','rumorTag','srcOrReact','id','lang']
    df_ = pd.DataFrame(columns=columns)
    dftxt_=pd.DataFrame(columns=txtdf)
    dir_path="/Users/monadiab/Trustworthiness/Data/all-rnr-annotated-threads/"
    lang_count={}
    for dirName, subdirList, fileList in os.walk(dir_path):
        for fname in fileList:
            if fname == "annotation.json":
                file_id = dirName.split("/")[-1]
                djson = os.path.join(dirName, fname)
                with open(djson) as f:
                    stc = json.load(f)
                if 'true' in stc.keys():
                    ver_rnr_tag[file_id]=str(stc['true'])
                else:
                    ver_rnr_tag[file_id]='nVer'

            if fname == "structure.json":
                file_id = dirName.split("/")[-1]
                djson = os.path.join(dirName, fname)
                with open(djson) as f:
                    stc = json.load(f)
                all_depth_and_max.update(mytree(stc, id=file_id)[0])

            if fname[-4:]=="json" and fname not in ["annotation.json","structure.json"] and fname not in old.keys() :
                if progCounter%500==0:
                    print (progCounter)
                progCounter+=1
                old[fname]=1
                djson=os.path.join(dirName,fname)
                parent = os.path.dirname(dirName).split('/')[-1]
                try:
                    smd = depth(subtree_finder(stc, fname[:-5])[fname[:-5]])
                except:
                    all_depth_and_max[fname[:-5]]=10000
                    smd=10000
                    print(fname[:-5])
                with open(djson) as f:
                    data = json.load(f)
                    if data['lang'] not in lang_count.keys():
                        lang_count[data['lang']]=1
                    else:
                        lang_count[data['lang']]+=1
                    if data['lang']!='en':
                        #only gets tweets which are english
                        continue
                    #############New
                    if "source-tweets" in dirName:
                        time_gap = 0
                        followers_count = data['user']['followers_count']
                        if 'media' in data['entities'].keys():
                            media = data['entities']['media']
                        else:
                            media = ""

                        urls = url_extraction(data, 'tw')
                        listed_count = data['user']['listed_count']
                        statuses_count = data['user']['statuses_count']
                        friends_count = data['user']['friends_count']
                        profile_image_url = data['user']['profile_image_url']
                        user_created_at = data['user']['created_at']
                        favourites_count = data['user']['favourites_count']
                        verified = str(data['user']['verified'])
                        user_url = data['user']['url']
                        description = data['user']['description']
                        time_zone = data['user']['time_zone']
                        ver_tag=ver_rnr_tag[parent]
                        # stford = stanfordHandler(data['text'])
                        #
                        # stford = stanfordHandler(data['text'])
                        hash = data['entities']['hashtags']
                        urlnurlfree = entityExtractor(data['text'])

                        post_created = data['created_at']
                        pos_time = time.mktime(time.strptime(post_created, "%a %b %d %H:%M:%S +0000 %Y")) + int(
                            data['user']['utc_offset'] or 0)

                        parent_body = urlnurlfree[1]
                        df_ = df_.append(
                            {'fname': fname[:-5], 'text': urlnurlfree[1], 'id': data['id'], 'parent_body': parent_body, \
                             'listed_count': listed_count, 'statuses_count': statuses_count,
                             'friends_count': friends_count, \
                             'profile_image_url': profile_image_url, 'user_created_at': user_created_at,
                             'favourites_count': favourites_count, 'verified': verified, \
                             'user_url': user_url, 'user_description': description, 'user_time_zone': time_zone, \
                             'created': pos_time, 'num_comment': data["retweet_count"], 'followers': followers_count, \
                             'ups': data['favorite_count'], \
                             'src_rtw': "src", 'user_json': data['user'],
                             'tree_max_depth': all_depth_and_max[parent + "_max"], 'media': media,\
                             'url': urls, 'Hashtag': hash, \
                             'veracitytag':ver_tag,\
                             'rum_tag':rumtagFinder(dirName),\
                             'depth': all_depth_and_max[fname[:-5]], \
                             'platform': 'twitter', 'parent_ids': parent, 'subnode_max_depth': smd, 'timeGap': time_gap,
                             'length': len(data['text'].split(" ")),'story':dirName.split("/")[6]}, ignore_index=True)
                        if parent not in parent_text.keys():
                            parent_text[parent] = data['text']
                    elif "reactions" in dirName:

                        listed_count = data['user']['listed_count']
                        statuses_count = data['user']['statuses_count']
                        friends_count = data['user']['friends_count']
                        profile_image_url = data['user']['profile_image_url']
                        user_created_at = data['user']['created_at']
                        favourites_count = data['user']['favourites_count']
                        verified = str(data['user']['verified'])
                        user_url = data['user']['url']
                        description = data['user']['description']
                        time_zone = data['user']['time_zone']
                        ver_tag = ver_rnr_tag[parent]

                        urls = url_extraction(data, 'tw')
                        followers_count = data['user']['followers_count']
                        if 'media' in data['entities'].keys():
                            media = data['entities']['media']
                        else:
                            media = ""
                        if parent not in parents_full_json:
                            src_path = "/".join(djson.split("/")[:-2]) + "/source-tweets/" + parent + ".json"
                            assert True == os.path.isfile(src_path);
                            "The path of the src generated wrongly" + src_path
                            with open(src_path) as srcf:
                                src_data = json.load(srcf)
                                parents_full_json[parent] = src_data
                                src_non_epoch = src_data['created_at']
                                src_created = time.mktime(
                                    time.strptime(src_non_epoch, "%a %b %d %H:%M:%S +0000 %Y")) + int(
                                    src_data['user']['utc_offset'] or 0)

                        else:
                            src_non_epoch = parents_full_json[parent]['created_at']
                            src_created = time.mktime(
                                time.strptime(src_non_epoch, "%a %b %d %H:%M:%S +0000 %Y")) + int(
                                src_data['user']['utc_offset'] or 0)
                        parent_body = parents_full_json[parent]['text']
                        post_created = data['created_at']
                        pos_time = time.mktime(time.strptime(post_created, "%a %b %d %H:%M:%S +0000 %Y")) + int(
                            src_data['user']['utc_offset'] or 0)
                        time_gap = (pos_time - src_created) / 3600

                        urlnurlfree = entityExtractor(data['text'])
                        # urls=data['entities']['urls']
                        hash = data['entities']['hashtags']
                        # stford = stanfordHandler(data['text'])
                        df_ = df_.append(
                            {'fname': fname[:-5], 'text': urlnurlfree[1], 'id': data['id'],
                             'listed_count': listed_count, 'statuses_count': statuses_count, 'parent_body': parent_body, \
                             'friends_count': friends_count, 'profile_image_url': profile_image_url, \
                             'user_created_at': user_created_at, 'favourites_count': favourites_count,
                             'verified': verified, \
                             'user_url': user_url, 'user_description': description, 'user_time_zone': time_zone, \
                             'created': pos_time, 'num_comment': data["retweet_count"], 'media': media,
                             'followers': followers_count, \
                             'ups': data['favorite_count'], 'Hashtag': hash,\
                             # 'sentiment': stford[2],'sent_dist': stford[1], 'pos': stford[0], \
                             'src_rtw': "ret", 'user_json': data['user'],
                             'tree_max_depth': all_depth_and_max[parent + "_max"], \
                             'veracitytag': ver_tag, 'depth': all_depth_and_max[fname[:-5]], 'url': urls, \
                             'rum_tag': rumtagFinder(dirName), 'platform': 'twitter',
                             'parent_ids': parent, 'subnode_max_depth': smd, 'timeGap': time_gap,
                             'length': len(data['text'].split(" ")),'story':dirName.split("/")[6]}, ignore_index=True)
                    else:
                        print(
                            "This file got error and has to be checked or the file is not in test or train list (exp. Some twitter data): " + fname)



                        ############Old
                    # df_=df_.append({'text':data['text'],'id':data['id'],'in_reply_to_status_id'\
                    #     :data['in_reply_to_status_id'],'story':dirName.split("/")[6],'entities':data['entities'],'user'\
                    #     :data['user'],'created_at':data['created_at'],'lang':data['lang'],'source':data['source'],\
                    #     'rumorTag':rumtagFinder(dirName),'srcOrReact':actionReactionFinder(dirName)}, ignore_index=True)
                    # dftxt_=dftxt_.append({'text':data['text'],'id':data['id'],'lang':data['lang'],\
                    #     'rumorTag':rumtagFinder(dirName),'srcOrReact':actionReactionFinder(dirName)}, ignore_index=True)
                    if dirName.split("/")[6] not in list_of_stories:
                        print("STOPPPPP")
            # print('Found directory: %s' % dirName)

    df_.to_pickle('data/Rumor/Rumor_all_meta.pkl')
    dftxt_.to_pickle('data/Rumor/rumDFText.pkl')

    print(df_.size)