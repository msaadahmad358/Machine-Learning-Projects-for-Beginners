#1 - import libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

import os

os.makedirs('outputs', exist_ok = True)
#2 - loading the dataset
data = pd.read_csv('dataset.csv')
X = data['text']
y = data['label']

#3 - splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#4 - text processing
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

#5 - training the nb classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

#6 - making predictions
y_pred = model.predict(X_test_vectorized)

#7 - evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy : {accuracy * 100}')
class_labels = np.unique(y_test)

plt.figure(figsize = (8,6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = class_labels, yticklabels =  class_labels)
plt.title('Confusion matrix Heatmap')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('outputs/1_confusion_matrix.jpg', dpi = 300, bbox_inches = 'tight')
plt.show()

#8 - prediction on unseen
user_input = ('I am good at football and play chess too.')
user_input_vectorized = vectorizer.transform([user_input])
predicted_label = model.predict(user_input_vectorized)
print(f'The input belongs to the {predicted_label[0]} category.')