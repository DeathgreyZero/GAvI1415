import nltk
import nltk.classify.util
import nltk.metrics

from nltk.corpus import stopwords


# Dato un insieme di tweets(training set) ritorna una lista
# contenente solo le parole.
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words


# Data una lista di parole, elimina le parole ripetute
# e ritorna una lista ordinata per frequenza di parola.
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


# Crea un dizionario che indica per ogni parola contenuta nel training set
# se e presente o meno nel tweet passato in input.
# esempio:
# {love: True, this: True, car: True, hate: False, Concert: False}
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


pos_tweets = [('I love this car', 'positive'),
              ('It will be easy', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('It will not be easy', 'negative'),
              ('I hate this car', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

if __name__ == "__main__":
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

    # Lista contenente le parole presenti nel training set ordinate per frequenza
    word_features = get_word_features(get_words_in_tweets(test_tweets))

    training_set = nltk.classify.apply_features(extract_features, test_tweets)
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    tweet = 'The beer is good'.lower()

    prob_t = classifier.prob_classify(extract_features(tweet.split()))

    print tweet
    print "Classificazione:", prob_t.max()
    print "Tweet positivo al", round(prob_t.prob('positive'), 2)*100, "%"
    print "Tweet negativo al", round(prob_t.prob('negative'), 2)*100, "%"