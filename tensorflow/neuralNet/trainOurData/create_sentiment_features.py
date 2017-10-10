
# coding:utf-8

"""
  [chair, table, spoon, television]
  I pull the chair up to the table.  =>  [1,1,0.0]  => chair and table appear one time, others are zero
  so [1,1,0,0] is a matrix to represent  the sentiment

"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as numpy
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000000


def create_lexicon(pos, neg):
	lexicon = []
	for fi in [pos, neg]:
		with open(fi,'r') as f:
			contents = f.readlines()
			for i in contents[:hm_lines]:
				all_words = word_tokenize(l)
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i n lexicon]

	w_counts = Counter(lexicon)







