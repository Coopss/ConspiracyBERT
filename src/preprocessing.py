from pymongo import MongoClient
# from auth import username, password
import urllib
from pprint import pprint
from tqdm import tqdm
import tensorflow as tf
import multiprocessing
from transformers import AutoTokenizer, TFAutoModel
import pickle
import os
import preprocessor as p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p.set_options(p.OPT.URL, p.OPT.EMOJI)

username = 'christian'
password = 'Dec211996'

client = MongoClient('mongodb://' + urllib.parse.quote_plus(username) + ':' + urllib.parse.quote_plus(password) + '@198.211.115.252')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# setup pickle data dir
data_dir = 'data'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def worker(consp_tuple):
    data = []
    conspiracy, hashtag = consp_tuple
    cursor = client[conspiracy][hashtag].find({})
    for i, document in enumerate(cursor):
        try:
            inputs = tokenizer(p.clean(document['text']), return_tensors="tf")
            inputs.update({'tweetId' : document['tweetId']})
            data.append(inputs)
        except Exception as e:
            pass


        if (i > 100):
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

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    results = p.map(worker, consp_tuples)

    return results

def standardize_tensor_shape():
    consp_tuples = get_consp_tuples()

    global_max = 0
    for consp_tuple in tqdm(consp_tuples):
        conspiracy, hashtag = consp_tuple
        try:
            with open(data_dir + '/' + conspiracy + '-' + hashtag + '.p', 'rb') as f:
                data = pickle.load(f)

                max_size = max(data, key= lambda x: x['input_ids'].shape[1])['input_ids'].shape[1]
                if max_size > global_max:
                    global_max = max_size

        except Exception as e:
            print("Could not find: " + data_dir + '/' + conspiracy + '-' + hashtag + '.p')
            pass

    print("global_max: " + str(global_max))

    for consp_tuple in tqdm(consp_tuples):
        conspiracy, hashtag = consp_tuple

        try:
            with open(data_dir + '/' + conspiracy + '-' + hashtag + '.p', 'rb') as f:
                data = pickle.load(f)

                input_ids = tf.zeros((0, global_max), dtype=tf.int32)
                token_type_ids = tf.zeros((0, global_max), dtype=tf.int32)
                attention_mask = tf.zeros((0, global_max), dtype=tf.int32)

                for row in data:
                    row_input_ids = tf.concat([row['input_ids'], tf.zeros((1,global_max-row['input_ids'].shape[1]), dtype=tf.int32)], axis=1)
                    row_token_type_ids = tf.concat([row['token_type_ids'], tf.zeros((1,global_max-row['token_type_ids'].shape[1]), dtype=tf.int32)], axis=1)
                    row_attention_mask = tf.concat([row['attention_mask'], tf.zeros((1,global_max-row['attention_mask'].shape[1]), dtype=tf.int32)], axis=1)

                    input_ids = tf.concat([input_ids, row_input_ids], axis=0)
                    token_type_ids = tf.concat([token_type_ids, row_token_type_ids], axis=0)
                    attention_mask = tf.concat([attention_mask, row_attention_mask], axis=0)

                conspiracy_x = {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}

            with open(data_dir + '/' + conspiracy + '-' + hashtag + '-standardized.p', 'wb') as f:
                pickle.dump(conspiracy_x, f)

        except Exception as e:
            print(e)
            print("Could not find: " + data_dir + '/' + conspiracy + '-' + hashtag + '.p')
            pass




    # for row in data:
    #     row['input_ids'] = tf.concat([row['input_ids'], tf.zeros((1,max_size-row['input_ids'].shape[1]), dtype=tf.int32)], axis=1)
    #     row['token_type_ids'] = tf.concat([row['token_type_ids'], tf.zeros((1,max_size-row['token_type_ids'].shape[1]), dtype=tf.int32)], axis=1)
    #     row['attention_mask'] = tf.concat([row['attention_mask'], tf.zeros((1,max_size-row['attention_mask'].shape[1]), dtype=tf.int32)], axis=1)

if __name__ == '__main__':
    print(preprocess())
    standardize_tensor_shape()