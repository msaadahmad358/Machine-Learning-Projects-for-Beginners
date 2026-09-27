#1 - import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('outputs', exist_ok = True)


#2 - load the dataset
data = pd.read_csv('dataset.csv')
print(data.head())
print(data.shape)

sns.countplot(x = 'label', data = data)
plt.savefig('outputs/1_class_distribution.jpg', dpi = 300, bbox_inches = 'tight')
plt.show()

#3 - balance the dataset
ham_msg = data[data['label'] == 'ham']
spam_msg = data[data['label'] == 'spam']

# down-sample ham emails to match the number of spam emails
ham_msg_balanced = ham_msg.sample(n = len(spam_msg), random_state = 42)

#combine balanced dataset
balanced_data = pd.concat([ham_msg_balanced,spam_msg]).reset_index(drop = True)

#visualize the balanced dataset
sns.countplot(x = 'label', data = balanced_data)
plt.title('Balanced Distributioon of Spam/Ham Emails')
plt.xticks(ticks = [0, 1], labels = ['Ham', 'Spam'])
plt.savefig('outputs/2_balanced_distributioon_of_emails', dpi = 300, bbox_inches = 'tight')
plt.show()

#4 - clean the text
balanced_data['text'] = balanced_data['text'].str.replace('Subject', ' ')
balanced_data.head()

punctuations_list = string.punctuation
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)
balanced_data['text'] = balanced_data['text'].apply(lambda x : remove_punctuations(x))
balanced_data.head()

#helper function that will help to remove the stopwords
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = []

    #storing the important words
    for word in str(text).split():
        word = word.lower()
        if word not in stop_words:
            imp_words.append(word)
    output = ' '.join(imp_words)
    return output

balanced_data['text'] = balanced_data['text'].apply(lambda text : remove_stopwords(text))
balanced_data.head()

#visualization word cloud
def plot_word_cloud(data, typ):
    email_corpus = ' '.join(data['text'])
    wc = WordCloud(background_color = 'white', max_words = 100, width = 800, height = 400).generate(email_corpus)
    plt.figure(figsize = (7,7))
    plt.imshow(wc, interpolation = 'bilinear')
    plt.title(f'Wordcloud for {typ} Emails', fontsize = 15)
    plt.axis('off')
    plt.savefig(f'outputs/3_wordcloud_{typ.lower()}.jpg', dpi = 300, bbox_inches = 'tight')
    plt.show()

plot_word_cloud(balanced_data[balanced_data['label'] == 'ham'], typ = 'Non-Spam')
plot_word_cloud(balanced_data[balanced_data['label'] == 'spam'], typ = 'Spam')

#6 - tokenization and padding
train_X, test_X, train_Y, test_Y = train_test_split(balanced_data['text'], balanced_data['label'], test_size = 0.2, random_state = 42)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)
train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)
max_len = 100 #maximum sequence length
train_sequences = pad_sequences(train_sequences, maxlen = max_len, padding = 'post', truncating = 'post')
test_sequences = pad_sequences(test_sequences, maxlen = max_len, padding = 'post', truncating = 'post')
train_Y = (train_Y == 'spam').astype(int)
test_Y = (test_Y == 'spam').astype(int)

#7 - define the model
model = tf.keras.models.Sequential([tf.keras.layers.Embedding(input_dim = len(tokenizer.word_index)+1, output_dim = 32, input_length = max_len), tf.keras.layers.LSTM(16), tf.keras.layers.Dense(32, activation = 'relu'), tf.keras.layers.Dense(1, activation = 'sigmoid')])
model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), optimizer = 'adam', metrics = ['accuracy'])
model.summary()

#8 - train the model
es = EarlyStopping(patience = 3, monitor = 'val_accuracy', restore_best_weights = True)
lr = ReduceLROnPlateau(patience = 2, monitor = 'val_loss', factor = 0.5, verbose = 0)
history = model.fit(train_sequences, train_Y, validation_data = (test_sequences, test_Y), epochs = 20, batch_size = 32, callbacks = [lr, es])

test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print('Test loss: ', test_loss)
print('Test accuracy: ', test_accuracy)

plt.plot(history.history['accuracy'], label = 'Training accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('outputs/4_model_accuracy.jpg', dpi = 300, bbox_inches = 'tight')
plt.show()