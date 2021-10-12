import datetime
import pickle
import time
import uuid

import dateutil.parser
import numpy as np
import regex as re
import requests

from log import getLogger
import pandas as pd

logger = getLogger()

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


def construct_dict_from_schema(schema, d):
    """ Recursively construct a new dictionary, using only fields from d that are in schema """
    new_dict = {}
    keys_deleted = []
    for row in schema:
        key_name = row['name']
        if key_name in d:
            # Handle nested fields
            if isinstance(d[key_name], dict) and 'fields' in row:
                new_dict[key_name] = construct_dict_from_schema(row['fields'], d[key_name])

            # Handle repeated fields - use the same schema as we were passed
            elif isinstance(d[key_name], list) and 'fields' in row:
                new_dict[key_name] = [construct_dict_from_schema(row['fields'], item) for item in d[key_name]]

            elif isinstance(d[key_name], str) and (str.upper(remove_punctuation(d[key_name])) == 'NULL' or remove_punctuation(d[key_name]) == ''):
                # don't add null values
                keys_deleted.append(key_name)
                pass

            elif not d[key_name] is None:
                if str.upper(row['type']) == 'TIMESTAMP':
                    # convert dates to datetimes
                    if not isinstance(d[key_name], datetime.datetime):
                        try:
                            _ts = None
                            if type(d[key_name]) == str:
                                if d[key_name].isnumeric():
                                    _ts = float(d[key_name])
                                else:
                                    new_dict[key_name] = dateutil.parser.parse(d[key_name])

                            if type(d[key_name]) == int or type(d[key_name]) == float or _ts:
                                if not _ts:
                                    _ts = d[key_name]

                                try:
                                    new_dict[key_name] = datetime.datetime.utcfromtimestamp(_ts)
                                except (ValueError, OSError):
                                    # time is likely in milliseconds
                                    new_dict[key_name] = datetime.datetime.utcfromtimestamp(_ts / 1000)

                            elif not isinstance(d[key_name], datetime.datetime):
                                new_dict[key_name] = pd.to_datetime(d[key_name])
                        except:
                            logger.error("Unable to parse {} item {}, type {}, into date format".format(key_name, d[key_name], type(d[key_name])))
                            #new_dict[key_name] = d[key_name]
                            pass
                    else:
                        # Already a datetime, move it over
                        new_dict[key_name] = d[key_name]
                elif str.upper(row['type']) == 'INTEGER':
                    # convert string numbers to integers
                    if isinstance(d[key_name],str):
                        try:
                            new_dict[key_name] = int(remove_punctuation(d[key_name]))
                        except:
                            logger.error("Unable to parse {} item {} into integer format".format(key_name, d[key_name]))
                            pass
                            #new_dict[key_name] = d[key_name]
                    else:
                        new_dict[key_name] = d[key_name]
                else:
                    new_dict[key_name] = d[key_name]
        else:
            keys_deleted.append(key_name)

    if len(keys_deleted)>0:
        logger.debug("Cleaned dict according to schema. Did not find {} keys: {}".format(len(keys_deleted),keys_deleted))

    set_orig = set(d.keys())
    set_new = set(new_dict.keys())
    set_removed = set_orig - set_new

    if len(set_removed)>0:
        logger.debug("Cleaned dict according to schema. Did not include {} keys: {}".format(len(set_removed), set_removed))

    return new_dict


def scrub_serializable(d):
    try:
        if isinstance(d, list):
            d = [scrub_serializable(x) for x in d]
            return d

        if isinstance(d, dict):
            for key in list(d.keys()):
                if d[key] is None:
                    del d[key]
                elif hasattr(d[key], 'dtype'):
                    d[key] = np.asscalar(d[key])
                elif isinstance(d[key], dict):
                    d[key] = scrub_serializable(d[key])
                elif isinstance(d[key], list):
                    d[key] = [scrub_serializable(x) for x in d[key]]
                elif isinstance(d[key], datetime.datetime):
                    # ensure dates are stored as strings in ISO format for uploading
                    d[key] = d[key].isoformat()
                elif isinstance(d[key], uuid.UUID):
                    # if the obj is uuid, we simply return the value of uuid
                    d[key] = d[key].hex

        return d
    except Exception as e:
        print(e)
        raise

def scrub_for_mongo(d):
    try:
        if isinstance(d, list):
            d = [scrub_for_mongo(x) for x in d]
            return d

        if isinstance(d, dict):
            for key in list(d.keys()):
                if d[key] is None:
                    del d[key]
                elif hasattr(d[key], 'dtype'):
                    # Ensure ints are stored as int64 etc
                    d[key] = np.asscalar(d[key])
                elif isinstance(d[key], dict):
                    d[key] = scrub_for_mongo(d[key])
                elif isinstance(d[key], list):
                    d[key] = [scrub_for_mongo(x) for x in d[key]]

        return d
    except Exception as e:
        print(e)
        raise


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


def fetch_with_backoff(session, url, wait_expo_max=900, max_tries=5, current_try=3):
    error_desc = ''
    if current_try > max_tries:
        raise IOError("Max retries failed fetching URL: {}.".format(url))
    try:
        r = session.get(url, timeout=31)

        if r.status_code == 429:
            error_desc = 'Hit rate limit.'
            pass
        else:
            return r
    except requests.exceptions.RequestException as e:
        error_desc = "Error fetching url: {}.\nError: {}".format(url, e)
        pass

    # Sleep for an exponentially increasing period.
    backoff = min(wait_expo_max, (2 ** current_try)) # + random.randint(0, 3)
    logger.error("{error_desc}. Sleeping for {backoff} seconds".format(error_desc=error_desc, backoff=backoff))
    time.sleep(backoff)
    return fetch_with_backoff(session, url, wait_expo_max, max_tries, current_try=current_try + 1)

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

