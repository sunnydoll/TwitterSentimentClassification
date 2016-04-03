# CS613 - Final Project
# Zhichao Cao
# zc77@drexel.edu

import csv, random
import nltk
import ner
import re
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
import string
from sklearn.svm import LinearSVC

# Extract features from list of words, with the frequency of occurring
def extractWordsFeatures(words):
	words = nltk.FreqDist(words)
	features = words.keys()
	return features

# Extract words from filtered tweets
def extractWordsFromTweets(filteredTweets):
	wordsList = []

	for(words, sentiment) in filteredTweets:
		wordsList.extend(words)

	return wordsList

# Further applying filter rule on tweets to decrease number of noisy words
def filterWords(tweet):
	pureTweets = []
	# words = tweet.split()
	words = [e.lower() for e in tweet.split() if "the" not in e and "and" not in e and len(e) >= 3 and "URL" not in e and "@" not in e]
	for word in words:
		word = filterDuplicate(word)
		word = word.strip('\'"?,.!')
		pureTweets.append(word.lower())
	return pureTweets

# Look for 2 or more repetitions of character and replace with the character itself
def filterDuplicate(ch):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", ch)


# Process the tweets, filter noisy words using pre-defined rules
def processTweet(tweet):	
	tweet = tweet.lower()
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
	tweet = re.sub('@[^\s]+','@',tweet)
	tweet = re.sub('[\s]+', ' ', tweet)
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	tweet = tweet.strip('\'"')
	# tweet = tweet.strip('\'"!?,.')
	return tweet

# Generate a feature dictionary
def generateFeatureDic(tweets):
	featuresDic = {}
	wordsSet = set(tweets)

	for word in featuresList:
		featuresDic[word] = (word in wordsSet)

	return featuresDic


# Entrance of executing program
# read all tweets and labels
trainfp = open( 'traindata.csv', 'rb' )
testfp = open( 'testdata.csv', 'rb' )
trainReader = csv.reader( trainfp, delimiter=',', quotechar='"', escapechar='\\' )
testReader = csv.reader( testfp, delimiter=',', quotechar='"', escapechar='\\' )

tweets = []
testTweets = []
filteredTweets = []

numPos = 0
numNeg = 0
numNeu = 0
numPost = 0
numNegt = 0
numNeut = 0

nbCurrect = 0
meCurrect = 0
svmCurrect = 0

for row in trainReader:
	sent = "neutral"
	if row[0] == '0':
		sent = "negative"
		numNeg = numNeg + 1
	elif row[0] == '4':
		sent = "positive"
		numPos = numPos + 1
	else:
		numNeu = numNeu + 1
	row[5] = processTweet(row[5])
	row[5] = filterWords(row[5])
	tweets.append( [row[5], sent] )
random.shuffle(tweets);

for row in testReader:
	sent = "neutral"
	if row[0] == '0':
		sent = "negative"
		numNegt = numNegt + 1
	elif row[0] == '4':
		sent = "positive"
		numPost = numPost + 1
	else:
		sent = "neutral"
		numNeut = numNeut + 1
	row[5] = processTweet(row[5])
	row[5] = filterWords(row[5])
	testTweets.append( [row[5], sent] )

trainfp.close()
testfp.close()

print "Total number of training tweets is " + str(len(tweets))
print "Number of positive tweets and negative tweets in training set is " + str(numPos) + " and " + str(numNeg) + " respectively."
print "Total number of testing tweets is " + str(len(testTweets))
print "Number of positive tweets and negative tweets in testing set is " + str(numPost) + " and " + str(numNegt) + " respectively."

exclude = set(string.punctuation)
# Applying rules in further step
for(words, sentiment) in tweets:
	# Using filter rules further
	words_filtered = words
	# words_filtered = [e.lower() for e in words.split() if "the" not in e and "at_user" not in e and len(e) >= 3 and "URL" not in e and "@" not in e]
	# for word in words_filtered:
		# word = word.strip('\'"?,.!:')
	filteredTweets.append((words_filtered, sentiment))

# Build a list of all distinct words ordered by frequency of occurring
featuresList = extractWordsFeatures(extractWordsFromTweets(filteredTweets))

# Build training dataset
trainingSet = nltk.classify.apply_features(generateFeatureDic, filteredTweets)

# Build Naive Bayes classifier using training dataset just created
nbClassifier = nltk.NaiveBayesClassifier.train(trainingSet)

# Build Maximum Entropy classifier using training dataset just created
meClassifier = nltk.classify.maxent.MaxentClassifier.train(trainingSet, 'GIS', trace=3, encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 10)

# Build Support Vector Machine with linear kernel classifier using training dataset just created
svmClassifier = nltk.classify.SklearnClassifier(LinearSVC())
svmClassifier.train(trainingSet)

# print nbClassifier.show_most_informative_features(32)

# Verify classifier using testing data set via three classifiers
for (words, sentiment) in testTweets:
	sent = sentiment
	res = nbClassifier.classify(generateFeatureDic(words))
	if res == sent:
		nbCurrect = nbCurrect + 1

for (words, sentiment) in testTweets:
	sent = sentiment
	res = meClassifier.classify(generateFeatureDic(words))
	if res == sent:
		meCurrect = meCurrect + 1

for (words, sentiment) in testTweets:
	sent = sentiment
	res = svmClassifier.classify(generateFeatureDic(words))
	if res == sent:
		svmCurrect = svmCurrect + 1

# Output accuracy
print "Accuracy under Naive Bayes is " + str(nbCurrect / (len(testTweets) * 1.0))
print "Accuracy under Maximum Entropy is " + str(meCurrect / (len(testTweets) * 1.0))
print "Accuracy under Support Vector Machine is " + str(svmCurrect / (len(testTweets) * 1.0))

