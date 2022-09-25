#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis: IMDB DATA SET

# In[18]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# #### READING DATA

# In[22]:


data = pd.read_csv('C:\\Users\\dell\\Desktop\\IMDB.csv')
data.head()


# In[23]:


data.info()


# #### Checking if we have any null values in the dataset

# In[24]:


data.isnull().sum()


# #### Converting Categorical Labels into Discrete

# In[25]:


data.replace(['positive', 'negative'], [1,0], inplace = True)


# In[26]:


data.head()


# ### Exploring Data

# #### Positive Review Example

# In[27]:


positive = data[data.sentiment == 1].sample(n = 1)['review'].iloc[0]
positive


# #### Negative Review Example

# In[28]:


negative = data[data.sentiment == 0].sample(n = 1)['review'].iloc[0]
negative


# In[29]:


fig, axs = plt.subplots(figsize = (15,7.5))

data['sentiment'].value_counts().plot.bar(color = ['red', 'green'])

plt.xticks(np.arange(2), ('Positive', 'Negative'), fontsize = 14)
axs.set_title('IMDB Reviews', fontsize = 18)
axs.set_xlabel('Sentiment', fontsize = 16)
axs.set_ylabel('Reviews Count', fontsize = 16)
axs.grid()
plt.show()


# In[30]:


data.sentiment.value_counts()


# #### 1) Removing HTML Tags using Regular Expression

# In[31]:


data.review


# #### Dealing with the stopwords

# In[32]:


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[33]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[34]:


stop_words = set(stopwords.words('english'))


# In[35]:


import re


# #### Preprocessing 

# ****Removing HTML tags****
# 
# ****Removing irrelevant characters****
# 
# ****Converting entire corpus into lower case****
# 
# ****Removing stopwords****
# 
# ****Converting entire dataset into lowercase****
# 
# ****Tokenization****
# 
# ****Normalization (Lemmatization)****

# In[36]:


nltk.download('punkt')


# In[37]:


preprocessed_review = []
for i in range(len(data.review)):
    cleaning = re.compile('<.*?>')                          # Pattern for removing html tags
    cleaning1 = re.compile('[^aA-zZ0-9]+')                  # Pattern for removing all the punctuations, commas, and other characters which can act as noise in data
    review = re.sub(cleaning, '', data.review.iloc[i])      # First substituing all the html tags with empty space
    review = re.sub(cleaning1, ' ', review)                 # Then substitutin all irrelevant characters with a single space
    
    review = review.lower()                                 # Converting all the text in reviews to lower case
    
    tokens = word_tokenize(review)                          # Splitting the reviews into individual words (tokenization)
    
    del review
    
    swords = []                                             # An empty list for storing all the words except stop words
    for word in tokens:
        if word not in stop_words:
            swords.append(word)
    del tokens
    
    lemmatizer = WordNetLemmatizer()                        # Normalization (Lemmatization)
    
    lemmas = []
    for lemma in swords:
        norm = lemmatizer.lemmatize(lemma)
        lemmas.append(norm)
    
    del swords
    
    lemmas = ' '.join(lemmas)
    preprocessed_review.append(lemmas)


# In[38]:


print(preprocessed_review[1])


# In[39]:


len(preprocessed_review)


# In[40]:


y = data['sentiment']
y.head()


# In[41]:


X = data.drop(['sentiment'], axis = 1)
X.head()


# In[42]:


X.shape


# ### Keras Text Tokenizer Vectorization

# In[43]:


import tensorflow as tf


# In[44]:


vocabulary = 15000
embedding = 20
max_length = 150
trunc_type = 'post'
oov_tok = ''


# In[45]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[46]:


tokenizer = Tokenizer(num_words = vocabulary, oov_token = oov_tok)
tokenizer.fit_on_texts(preprocessed_review)
word_index = tokenizer.word_index
word_index


# In[47]:


sequences = tokenizer.texts_to_sequences(preprocessed_review)
padded = pad_sequences( sequences , maxlen = max_length , truncating = trunc_type)


# In[48]:


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(padded, y, test_size = 0.2, random_state = 120, shuffle = True)
print(f'shape of training data is: {xtrain.shape, ytrain.shape}')
print(f'shape of testing data is: {xtest.shape, ytest.shape}')


# #### RNN
# 
# 

# In[49]:


from tensorflow.keras.layers import SimpleRNN , BatchNormalization, Dense , Embedding
from tensorflow.keras import Sequential


# In[50]:


model = Sequential()
model.add(Embedding(vocabulary , embedding , input_length = max_length ))
model.add(SimpleRNN(32))
model.add(Dense(10))
model.add(Dense(1 , activation = 'sigmoid'))
model.summary()


# In[51]:


# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[52]:


history = model.fit(xtrain, ytrain, validation_split = (0.1) , epochs=,  batch_size = 1024 , verbose = 1)


# In[53]:


model.evaluate(xtest,ytest)


# In[54]:


predictions = np.argmax(model.predict(xtest),axis = 1)
predictions


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix ,classification_report
cm=confusion_matrix(ytest,predictions)
print(cm)


# In[ ]:


print(accuracy_score(ytest,predictions))


# In[ ]:


import seaborn as sns

fig = plt.figure(figsize=(10,7))
confusion_matrix = cm
sns.heatmap(confusion_matrix,annot = True,fmt = 'd')


# In[ ]:


print(classification_report(ytest,predictions))


# #### NAIVE BAYES CLASSIFICATION

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

naive = MultinomialNB()
naive.fit(xtrain, ytrain)
predictions = naive.predict(xtest)
predictions


# In[ ]:


accuracy = accuracy_score(ytest, predictions)
print(f'The accuracy of our model: {accuracy}')


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# In[ ]:


naive_report = classification_report(ytest, predictions)
print(naive_report)


# In[ ]:


naive_cm = confusion_matrix(ytest, predictions)
fig,axs = plt.subplots(figsize = (5,3.5), dpi = 150)
sns.heatmap(naive_cm, annot = True, fmt = 'd', cmap="YlGnBu")
axs.set_xlabel('Predicted Target', fontsize = 14)
axs.set_ylabel('Actual Target', fontsize = 14)
axs.set_title('Confusion Matrix', fontsize = 16)
plt.style.use('dark_background')


# In[45]:


from sklearn.linear_model import LogisticRegression

logic = LogisticRegression()
logic.fit(xtrain, ytrain)
logic_pred = logic.predict(xtest)
print(logic_pred)
logic_acc = accuracy_score(ytest, logic_pred)
print(f'Logistic Regression Gave an accuracy  of {logic_acc}')


# In[46]:


logic_report = classification_report(ytest, logic.predict(xtest))
print(logic_report)


# In[50]:


logic_cm = confusion_matrix(ytest,logic_pred)
fig,axs = plt.subplots(figsize = (5,3.5), dpi = 150)
sns.heatmap(logic_cm, annot = True, fmt = 'd', cmap="Reds")
axs.set_xlabel('Predicted Target', fontsize = 14)
axs.set_ylabel('Actual Target', fontsize = 14)
axs.set_title('Confusion Matrix', fontsize = 16)

