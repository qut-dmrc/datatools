import os

import base64

from hashlib import sha256

import datetime
import hmac
import pickle
import time
import uuid

import backoff as backoff
import dateutil.parser
import numpy as np
import regex as re
import requests

from datatools.log import getLogger
import pandas as pd

logger = getLogger()

ENV_HMAC_KEY = 'HMAC_KEY'

def hmac_sha256(identifier):
    """ Convert an identifier to a pseudonymous hash.
    We use HMAC with SHA256 to hash identifiers. This allows us to retain referential integrity without
    storing personally identifiable information. We use a secret key to avoid dictionary attacks.
    """

    if not identifier:
        return None

    key = os.environ[ENV_HMAC_KEY]
    assert key
    key = key.encode()

    if isinstance(identifier, bytes):
        pass
    elif isinstance(identifier, str):
        identifier = identifier.encode()
    else:
        identifier = str(identifier).encode()


    h = hmac.new(key, identifier, sha256)
    encoded_id = base64.b64encode(h.digest()).decode()
    return encoded_id

def remove_punctuation(text):
    return re.sub(r"\p{P}+", "", text)


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    # remove punctuation
#        stems = [remove_punctuation(s) for s in stems]
    stems = map(remove_punctuation, stems)

    # only works for ascii punctuation
    #stems = [''.join(c for c in s if c not in string.punctuation) for s in stems]
    # remove empty strings
    stems = [s for s in stems if s]

    return stems



def lemmatize(df, text_column_name, lemma_column_name):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    df[lemma_column_name] = None
    #df[lemma_column_name] = df[text_column_name].apply(remove_punctuation).str.lower().str.split()

    wordnet_tag = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}

    for index, row in df.iterrows():
        tokens = nltk.word_tokenize(remove_punctuation(row[text_column_name]).lower())
        tagged = nltk.pos_tag(tokens)
        lemmas = []
        for t in tagged:
            try:
                lemmas.append(lemmatizer.lemmatize(t[0], wordnet_tag[t[1][:2]]))
            except:
                lemmas.append(lemmatizer.lemmatize(t[0]))
        df.set_value(index, lemma_column_name, lemmas)

#    df[lemma_column_name].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    return df


def convert_timestamp(str):
    ts = time.strptime(str, '%a %b %d %H:%M:%S +0000 %Y')
    ts = time.strftime('%Y-%m-%d %H:%M:%S', ts)

    return ts

def twitter_scrub(d):
    # removes unnecessary info from a tweet object
    # also check for extended tweets and rewrite the status

    try:
        d['text'] = d['full_text']
    except KeyError:
        d['text'] = d.get('extended_tweet', {}) \
            .get('full_text', d.get('text', None))



    hashtags = []
    try:
        for h in d['entities']['hashtags']:
            hashtags.append(h.get('text'))
    except:
        pass
    try:
        for h in d['retweeted_status']['entities']['hashtags']:
            hashtags.append(h.get('text'))
    except: pass
    try:
        for h in d['quoted_status']['entities']['hashtags']:
            hashtags.append(h.get('text'))
    except:
        pass
    d['hashtags'] = hashtags


    urls = []
    try:
        for x in d['entities']['urls']:
            urls.append(x.get('expanded_url'))
    except:
        pass
    try:
        for x in d['retweeted_status']['entities']['urls']:
            urls.append(x.get('expanded_url'))
    except:
        pass
    try:
        for x in d['quoted_status']['entities']['urls']:
            urls.append(x.get('expanded_url'))
    except:
        pass
    d['urls'] = urls


    mentions = []
    try:
        for m in d['entities']['user_mentions']:
            mention = {}
            mention['type'] = 'mention'
            mention['user_id'] = m.get('id', None)
            mentions.append(mention)
    except:
        pass

    try:  # retweet
        mention = {}
        mention['type'] = 'retweet'
        mention['user_id'] = d['retweeted_status']['user']['id']
        mention['status_id'] = d['retweeted_status']['id']
        try:
            mention['text'] = d['retweeted_status']['full_text']
        except KeyError:
            mention['text'] = d['retweeted_status'].get('extended_tweet', {}) \
                .get('full_text', d['retweeted_status'].get('text', None))
        mentions.append(mention)

        try:
            for m in d['retweeted_status']['entities']['user_mentions']:
                mention = {}
                mention['type'] = 'retweet mention'
                mention['user_id'] = m.get('id', None)
                mentions.append(mention)
        except:
            pass
    except:
        pass

    try:  # quote
        mention = {}
        mention['type'] = 'quote'
        mention['user_id'] = d['retweeted_status']['quoted_status']['user']['id']
        mention['status_id'] = d['retweeted_status']['quoted_status']['id']
        try:
            mention['text'] = d['retweeted_status']['quoted_status']['full_text']
        except KeyError:
            mention['text'] = d['retweeted_status']['quoted_status'].get('extended_tweet', {}) \
                .get('full_text', d['retweeted_status']['quoted_status'].get('text', None))
        mentions.append(mention)

        try:
            for m in d['retweeted_status']['quoted_status']['entities']['user_mentions']:
                mention = {}
                mention['type'] = 'quote mention'
                mention['user_id'] = m.get('id', None)
                mentions.append(mention)
        except:
            pass
    except:
        pass


    try:  # retweeted quote
        mention = {}
        mention['type'] = 'retweeted quote'
        mention['user_id'] = d['quoted_status']['user']['id']
        mention['status_id'] = d['quoted_status']['id']
        try:
            mention['text'] = d['quoted_status']['full_text']
        except KeyError:
            mention['text'] = d['quoted_status'].get('extended_tweet', {}) \
                .get('full_text', d['quoted_status'].get('text', None))
        mentions.append(mention)

        try:
            for m in d['quoted_status']['entities']['user_mentions']:
                mention = {}
                mention['type'] = 'retweeted quote mention'
                mention['user_id'] = m.get('id', None)
                mentions.append(mention)
        except:
            pass
    except:
        pass

    d['mentions'] = mentions

    media = []
    try:
        for m in d['extended_entities']['media'], []:
            media.append(m.get('media_url'))
    except:
        pass
    try:
        for m in d['retweeted_status']['extended_entities']['media']:
            media.append(m.get('media_url'))
    except:
        pass
    try:
        for m in d['retweeted_status']['extended_entities']['media']:
            media.append(m.get('media_url'))
    except:
        pass

    d['media'] = media

    return d  # For convenience

from math import log2

_suffixes = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']


def file_size(size):
    """ determine binary order in steps of size 10
    From https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    """
    # (coerce to int, // still returns a float)
    order = int(log2(size) / 10) if size else 0
    # format file size
    # (.4g results in rounded numbers for exact matches and max 3 decimals,
    # should never resort to exponent values)
    return '{:.4g} {}'.format(size / (1 << (order * 10)), _suffixes[order])


def pickle_rows(result_rows, file_name):
    """ Try to save the result rows. Sometimes it doesn't work - try three times then fail."""

    success = False
    for i in range(0, 2):
        try:
            with open(file_name, 'wb') as handle:
                pickle.dump(result_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info("Saved {}.".format(file_name))
            success = True
            break
        except Exception as e:
            logger.exception("Failed saving {}, attempt {}, reason: ".format(file_name, i, e))
            pass

    return success


def unix_time_millis(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)

    return (dt - epoch).total_seconds() * 1000.0


class TooManyRequests(Exception):
    """Too many requests"""
    pass


@backoff.on_exception(backoff.expo,
                      (requests.exceptions.RequestException, TooManyRequests),
                      max_tries=5)
def fetch_with_backoff(session, url, **kwargs):
    r = session.get(url)
    r.raise_for_status()
    return r

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def find_first_key_in_dict(x, search_key):
    if isinstance(x, dict):
        if search_key in x:
            return x[search_key]
        else:
            for k in x:
                result = find_first_key_in_dict(x[k], search_key)
                if result:
                    return result
    elif isinstance(x, list):
        for y in x:
            result = find_first_key_in_dict(y, search_key)
            if result:
                return result

    return False


def find_all_keys_in_dict(x, search_key):
    results = []

    if isinstance(x, dict):
        if search_key in x and x[search_key]:
            if isinstance(x[search_key], list):
                results.extend(x[search_key])
            else:
                results.append(x[search_key])
        else:
            for k in x:
                result = find_all_keys_in_dict(x[k], search_key)
                if result:
                    results.extend(result)
    elif isinstance(x, list):
        for y in x:
            result = find_all_keys_in_dict(y, search_key)
            if result:
                results.extend(result)

    return results

def safe_sample_df(df, max_sample_size):
    try:
        df = df.sample(max_sample_size)
    except ValueError:
        pass  # usually because there weren't enough records
    return df