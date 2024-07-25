# -*- coding: utf-8 -*-
"""project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NelcS5XYzqQcVq9CvSyKtKjWLD6l-vkS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/gdrive')
# %cd '/content/gdrive/MyDrive/ML'

df= pd.read_csv("LD.csv")
df

df.shape

df.drop(df[df.duplicated()].index, axis=0, inplace=True)
df.shape

df["Language"].nunique()

#check xem có bao nhiêu ngôn ngữ được train
df["Language"].value_counts()

plt.figure(figsize=(12, 6))
ax = sns.countplot(x='Language', data=df, palette='viridis', order=df['Language'].value_counts().index)
plt.title('Distribution of Languages')
plt.xticks(rotation=45, ha='right')

total = len(df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height() + 20
    ax.text(x, y, percentage, ha='center', fontsize=8)
plt.show()

data= df.copy()
data['Cleaned_Text']= ""
data

import re
def clean_function(Text):
    # xóa các kí tự số và các kí tự đặc biệt
    Text = re.sub(r'[\([{})\]!@#$,"%^*?:;~`0-9]', ' ', Text)

    Text = Text.lower()
    Text = re.sub('http\S+\s*', ' ', Text)
    Text = re.sub('RT|cc', ' ', Text)
    Text = re.sub('#\S+', '', Text)
    Text = re.sub('@\S+', '  ', Text)
    Text = re.sub('\s+', ' ', Text)

    return Text

data['Cleaned_Text'] = data['Text'].apply(lambda x: clean_function(x))
data

X= data["Cleaned_Text"]
y= data["Language"]

from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
y= encoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

# Vector hóa văn bảng
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

"""Mô hình Naive Bayes"""

param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]  # Tuning the smoothing parameter
}

# Tạo GridSearchCV
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Classification Report:\n{classification_report_str}')

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print('\n' + '='*50 + '\n')

"""Mô hình RandomForest"""

param_grid = {
    'n_estimators': [50, 100, 200],  # Số cây trong rừng
    'max_depth': [None, 10, 20, 30],  # Độ sâu tối đa của cây
    'min_samples_split': [2, 5, 10]  # Số mẫu tối thiểu cần thiết để chia nút
}

# Khởi tạo GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_tfidf)

# Đánh giá mô hình với các chỉ số bổ sung
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Classification Report:\n{classification_report_str}')

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print('\n' + '='*50 + '\n')

"""Mô hình CNN"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Xây dựng mô hình CNN
best_model = Sequential()
best_model.add(Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)))
best_model.add(Dropout(0.5))
best_model.add(Dense(64, activation='relu'))
best_model.add(Dropout(0.5))
best_model.add(Dense(len(encoder.classes_), activation='softmax')) #dua chuoi ve dung so nhan ban dau de kiem tra

best_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = best_model.fit(X_train_tfidf, y_train, epochs=10, batch_size=4, validation_split=0.2)
loss, accuracy = best_model.evaluate(X_test_tfidf, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

"""Hàm in cho 2 model naive bayes và randomForest"""

# hàm in ra dự đoán
def predict(text):
    text_tfidf = tfidf_vectorizer.transform([text]).toarray()
    lang = best_model.predict(text_tfidf)
    language = encoder.inverse_transform(lang)[0]
    print('The Language is in', language)

"""Hàm in cho CNN"""

def predict(text):
    text_tfidf = tfidf_vectorizer.transform([text]).toarray()  # Chuyển sang dạng TF-IDF
    predictions = best_model.predict(text_tfidf)  # Dự đoán nhãn
    predicted_class = np.argmax(predictions, axis=1)
    language = encoder.inverse_transform(predicted_class)  # Chuyển đổi chỉ số thành nhãn
    return language[0]

# English
predict("Hello iam a boy")