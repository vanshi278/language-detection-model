# %%
print("hello")


# %%
import string
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns



# %%
df = pd.read_csv('Language Detection.csv')
df.head()


# %%
string.punctuation


# %%


# %%
def remove_pun(text):
    for pun in string.punctuation:
        text = str(text)  
        text = text.replace(pun," ")
    text = text.lower()
    return (text)
    

# %%
print(df['Text'].iloc[0])
print(type(df['Text'].iloc[0]))

# %%
df['Text'] = df['Text'].apply(remove_pun)


# %%
df.head()


# %%
from sklearn.model_selection import train_test_split

# %%
X = df.iloc[:,0]
Y = df.iloc[:,1]

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2)


# %%



# %%
from sklearn import feature_extraction


# %%
vec = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), analyzer='char')

# %%
from sklearn import pipeline
from sklearn import linear_model

# %%
model_pipe= pipeline.Pipeline([('vec', vec),('clf', linear_model.LogisticRegression())]) 

# %%
model_pipe.fit(X_train, Y_train)

# %%
predict_val = model_pipe.predict(X_test)

# %%
from sklearn import metrics


# %%
metrics.accuracy_score(Y_test, predict_val)*100


# %%
metrics.confusion_matrix(Y_test, predict_val)

# %%
import pickle


# %%
new_file = open('model.pckl', 'wb')
pickle.dump(model_pipe, new_file)
new_file.close()

# %%
import os


# %%





# %%



