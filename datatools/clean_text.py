import string
import re
import string
from collections import Counter
from itertools import chain


def remove_punctuation(text):
    return re.sub(r"\p{P}+", "", text)


class RemovePuncUnicode():
    def __init__(self):
        punc = u''
        tbl = {}
        for i in xrange(sys.maxunicode):
            char = unichr(i)
            if unicodedata.category(char).startswith('P'):
                tbl[i] = None
                punc += char
        self.exclude = set(punc)

    def strip_punctuation(self, unicode_string):
        return ''.join([ch for ch in unicode_string if ch not in self.exclude])

class cleaner():
    def __init__(self, stopwords):
        self.stop_words_list = stopwords

    # Identify most common words - to consider adding to stopwords below.
    # returns the counter object for later processing
    def count_words(documents):
        c = Counter(chain.from_iterable(documents))
        # use c.most_commmon(n) to get a list of the most common words and their counts.
        return c

    def keep_nouns(documents):
        # Take a list of lists, and remove everything that is not a noun
        # Return a list of lists.

        newlist = []

        for text in documents:
            tagged = nltk.pos_tag(text)  # use NLTK's part of speech tagger

            keep = ['NN', 'NNS', 'NNP', 'NNPS']  # keep only nouns
            text = [word for word, pos in tagged if pos in keep]
            newlist.append(text)

        return newlist

    #    def stop_words_list():
    #        '''
    #            A stop list specific to the observed timelines composed of noisy words
    #            This list would change for different set of timelines
    #        '''
    #        return ['news','data', 'retention','metadata']
    #        #return ['uber', '#uber', 'rt', "it's","get","i'm","via","amp","thi","driver","taxi","ride"]

    def all_stopwords(self, documents):
        '''
            Builds a stoplist composed of stopwords in several languages,
            tokens with one or 2 words and a manually created stoplist
        '''
        # tokens with 1 characters
        unigrams = [item for sublist in documents for item in sublist if len(item) == 1]
        bigrams = [item for sublist in documents for item in sublist if len(item) == 2]

        # Compile global list of stopwords
        stoplist = set(nltk.corpus.stopwords.words("english")
                       #                    + nltk.corpus.stopwords.words("french")
                       #                    + nltk.corpus.stopwords.words("german")
                       + self.stop_words_list
                       + unigrams + bigrams
                       )
        return stoplist

    def stem(self, documents):
        # take a list of lists, and convert words to stems
        # returns a list of lists of stemmed words
        from nltk.stem.porter import PorterStemmer

        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()

        # stem token
        stemmed_list = [[p_stemmer.stem(word) for word in document] for document in documents]

        return stemmed_list

    def remove_rare(self, documents, count_of_words):
        # remove words that appear only once
        # expects a Counter like c = Counter(chain.from_iterable(documents))
        texts = [[word for word in x if count_of_words[word] > 1] for x in documents]

        return texts

    def clean(self, documents, stoplist):
        # Accepts list of lists
        # Returns cleaned list of lists

        cleaned_list = []
        for text in documents:
            # remove punctuation
            text = [''.join(c for c in s if c not in string.punctuation) for s in text]
            # remove empty strings
            text = [s for s in text if s]

            # remove links
            text = [word for word in text if word[:4] != 'http']

            cleaned_list.append(text)

        # stem the list
        cleaned_list = self.stem(cleaned_list)

        # remove common words
        #stoplist = self.all_stopwords(cleaned_list)
        cleaned_list = [[word for word in document if word not in stoplist] for document in cleaned_list]

        # use the builtin dictionary.filter_extremes() for this instead. See below.
        # c = count_words(cleaned_list)
        # remove words that appear only once, and words that appear very often
        # cleaned_list = remove_rare(cleaned_list, c)


        return cleaned_list

        # see here for example lemmatize: http://blog.cigrainger.com/tag/python-lda-gensim.html
        #    return ' '.join(
        #        lmtzr.lemmatize(word, get_wordnet_pos(tag[1]))
        #        for word, tag in zip(filtered_words, tags)
        #    )


''' OTHER potential things to do: https://gist.github.com/a-paxton/40fd496bba6edcb8cb87


# remove characters and stoplist words, then generate dictionary of unique words
data['text_data'].replace('[!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]',' ',inplace=True,regex=True)
wordlist = filter(None, " ".join(list(set(list(itertools.chain(*data['text_data'].str.split(' ')))))).split(" "))
data['stemmed_text_data'] = [' '.join(filter(None,filter(lambda word: word not in stop, line))) for line in data['text_data'].str.lower().str.split(' ')]

# remove all words that don't occur at least 5 times and then stem the resulting docs
minimum_count = 5
str_frequencies = pd.DataFrame(list(Counter(filter(None,list(itertools.chain(*data['stemmed_text_data'].str.split(' '))))).items()),columns=['word','count'])
low_frequency_words = set(str_frequencies[str_frequencies['count'] < minimum_count]['word'])
data['stemmed_text_data'] = [' '.join(filter(None,filter(lambda word: word not in low_frequency_words, line))) for line in data['stemmed_text_data'].str.split(' ')]
data['stemmed_text_data'] = [" ".join(stemmer.stemWords(re.sub('[!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]',' ', next_text).split(' '))) for next_text in data['stemmed_text_data']]

'''


