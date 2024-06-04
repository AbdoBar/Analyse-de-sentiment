import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')

import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Step 1: Load the dataset تحميل البيانات
df = pd.read_csv('dataset.csv')

# Step 2: Data cleaning
df['التعليق'] = df['التعليق'].astype(str) #تجهيز النص 
df['التعليق'] = df['التعليق'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Step 3: Text normalization

# Step 4: Tokenization تقسيم النص الى كلمات
df['tokens'] = df['التعليق'].apply(lambda x: word_tokenize(x))

# Step 5: Stop-word removal حذف الكلمات التي لا تأثر
stop_words = set(stopwords.words('arabic'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# Step 6: Word embedding تحويل النص الى ارقام
model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1)
word_vectors = model.wv

# Step 7: Padding
max_seq_length = 100
df['padded_tokens'] = df['tokens'].apply(lambda x: pad_sequences([[int(token) for token in x if token.isdigit()]], maxlen=max_seq_length, padding='post')[0])


# Step 8: Train-test split تقسيم البيانات
X = df['padded_tokens'].tolist()
y = df['الرقم'].tolist()
y = to_categorical(y)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC


svm_model = SVC()


svm_model.fit(X_train, y_train.argmax(axis=1))  


y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test.argmax(axis=1), y_pred) 
precision = precision_score(y_test.argmax(axis=1), y_pred)
recall = recall_score(y_test.argmax(axis=1), y_pred)
f1score = f1_score(y_test.argmax(axis=1), y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1score)


from sklearn.linear_model import LogisticRegression


logreg_model = LogisticRegression()

logreg_model.fit(X_train, y_train.argmax(axis=1))


y_pred = logreg_model.predict(X_test)


accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)
precision = precision_score(y_test.argmax(axis=1), y_pred)
recall = recall_score(y_test.argmax(axis=1), y_pred)
f1score = f1_score(y_test.argmax(axis=1), y_pred)
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall  :", recall)
print("F1 Score :", f1score)


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train.argmax(axis=1)) 


y_pred = rf_model.predict(X_test)


accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)
precision = precision_score(y_test.argmax(axis=1), y_pred)
recall = recall_score(y_test.argmax(axis=1), y_pred)
f1score = f1_score(y_test.argmax(axis=1), y_pred)
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1score)

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()


nb_model.fit(X_train, y_train.argmax(axis=1))  


y_pred = nb_model.predict(X_test)


accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)  
precision = precision_score(y_test.argmax(axis=1), y_pred)
recall = recall_score(y_test.argmax(axis=1), y_pred)
f1score = f1_score(y_test.argmax(axis=1), y_pred)
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1score)


from sklearn.tree import DecisionTreeClassifier


dt_model = DecisionTreeClassifier()


dt_model.fit(X_train, y_train.argmax(axis=1)) 


y_pred = dt_model.predict(X_test)


accuracy = accuracy_score(y_test.argmax(axis=1), y_pred) 
precision = precision_score(y_test.argmax(axis=1), y_pred)
recall = recall_score(y_test.argmax(axis=1), y_pred)
f1score = f1_score(y_test.argmax(axis=1), y_pred)
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1score)



from sklearn.neighbors import KNeighborsClassifier


knn_model = KNeighborsClassifier()


knn_model.fit(X_train, y_train.argmax(axis=1))  


y_pred = knn_model.predict(X_test)


accuracy = accuracy_score(y_test.argmax(axis=1), y_pred) 
precision = precision_score(y_test.argmax(axis=1), y_pred)
recall = recall_score(y_test.argmax(axis=1), y_pred)
f1score = f1_score(y_test.argmax(axis=1), y_pred)
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1score)


