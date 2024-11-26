# Install required libraries
#!pip install nltk tensorflow dialogflow
#!pip install dialogflow
!pip install google-cloud-dialogflow
#pip install nltk tensorflow google-cloud-dialogflow
#!pip install ntlk tensorflow dialogflow
import nltk
import random
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from google.cloud import  dialogflow