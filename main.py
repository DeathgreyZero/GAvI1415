import nltk, nltk.classify.util, nltk.metrics
from nltk.corpus import stopwords

# Ritorna una lista contenente tutte le parole
# contenute nel training set
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

pos_tweets = [('I love this car', 'positive'),
              ('It will be easy','positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('It will not be easy','negative'),
              ('I hate this car','negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

# Creazione di una lista contenente un array di parole prive di stopwords
# con accanto la classificazione positivo/negativo
wnl = nltk.WordNetLemmatizer()

test_tweets = []
for (phrase, sentiment) in pos_tweets + neg_tweets:
    tokens = nltk.word_tokenize(phrase)
    for t in tokens:
        if t in stopwords.words('english'):
            tokens.remove(t)
    test_tweets.append((tokens, sentiment))

print test_tweets
print get_words_in_tweets(test_tweets)
print get_word_features(get_words_in_tweets(test_tweets))

word_features = get_word_features(get_words_in_tweets(test_tweets))

training_set = nltk.classify.apply_features(extract_features, test_tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

tweet = 'I do not really love this work'.lower()

print tweet
print classifier.classify(extract_features(tweet.split()))