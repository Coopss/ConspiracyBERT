import os
from pymongo import MongoClient
import urllib
import multiprocessing
import pickle
import random

import preprocessor as p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
p.set_options(p.OPT.URL, p.OPT.EMOJI)
data_dir = '../scripts/data'

username = 'christian'
password = 'Dec211996'

client = MongoClient('mongodb://' + urllib.parse.quote_plus(username) + ':' + urllib.parse.quote_plus(password) + '@198.211.115.252')

def worker(consp_tuple):
    data = []

    conspiracy, hashtag = consp_tuple
    cursor = client[conspiracy][hashtag].find({})
    for i, document in enumerate(cursor):
        try:
            inputs = {'text' : p.clean(document['text'])}
            inputs.update({'tweetId' : document['tweetId']})
            data.append(inputs)
        except Exception as e:
            pass


        if (i > 1000):
            break

    with open(data_dir + '/' + conspiracy + '-' + hashtag + '.p', 'wb') as f:
        pickle.dump(data, f)

    return consp_tuple

def get_consp_tuples():
    ignore_mask = ['test_db', 'admin', 'local', 'config', 'TwitterJobs']
    conspiracies = list(set(client.list_database_names()) - set(ignore_mask))

    consp_tuples = []
    for conspiracy in conspiracies:
        for hashtag in client[conspiracy].list_collection_names():
            consp_tuples.append((conspiracy, hashtag))

    return consp_tuples

def preprocess():
    consp_tuples = get_consp_tuples()
    random.shuffle(consp_tuples)
    p = multiprocessing.Pool(multiprocessing.cpu_count())

    results = p.map(worker, consp_tuples)

if __name__ == '__main__':

    preprocess()