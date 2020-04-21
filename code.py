# --------------
import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Code Starts Here
df=pd.read_csv(path_train)
display(df.head())

#The function label_race checks every row for a category that is marked as T and return the name of the first category.
def label_race(row):
  for category in row.keys():
    if row[category]=='T':
      return category

#Create a new column category which contains the values obtained by applying to function to rows
df['category']=df.apply(lambda x : label_race(x[df.columns!='message']),axis=1)

# Drop the columns of food, recharge, support, reminders, nearby, movies, casual, other and travel 
df.drop(columns=['food', 'recharge', 'support', 'reminders', 'nearby', 'movies', 'casual', 'other', 'travel'],inplace=True)

display(df.head())




# --------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# Sampling only 1000 samples of each category
df = df.groupby('category').apply(lambda x: x.sample(n=1000, random_state=0))

# Code starts here
all_text = df.message.apply(lambda x : x.lower())
tfidf=TfidfVectorizer(stop_words='english')
tfidf.fit(all_text)
X=tfidf.transform(all_text).toarray()
le=LabelEncoder()
le.fit(df.category)
y=le.transform(df.category)
print(X.shape,y.shape)


# --------------
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Code starts here
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.3, random_state=42)
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_val)
log_accuracy =accuracy_score(y_val,y_pred)
# Instantiate MultinomialNB model  
nb=MultinomialNB()

# Fit on X_train and y_train
nb.fit(X_train,y_train)

# Predict the values using the above fitted model for X_val
y_pred=nb.predict(X_val)

# Calculate the accuracy score
nb_accuracy=accuracy_score(y_val,y_pred)
print('MultinomialNB Accuracy Score',nb_accuracy)

# Instantiate LinearSVC
lsvm=LinearSVC(random_state=0)

# Fit on X_train and y_train
lsvm.fit(X_train,y_train)

# Predict the values using the above fitted model for X_val
y_pred=lsvm.predict(X_val)

# Calculate the accuracy score
lsvm_accuracy=accuracy_score(y_val,y_pred)
print('LinearSVC Accuracy Score',lsvm_accuracy)


# --------------
# path_test : Location of test data

#Loading the dataframe
df_test = pd.read_csv(path_test)

#Creating the new column category
df_test["category"] = df_test.apply (lambda row: label_race (row),axis=1)

#Dropping the other columns
drop= ["food", "recharge", "support", "reminders", "nearby", "movies", "casual", "other", "travel"]
df_test=  df_test.drop(drop,1)

# Code starts here

#Loading the dataframe
df_test = pd.read_csv(path_test)

#Creating the new column category
df_test["category"] = df_test.apply (lambda row: label_race (row),axis=1)

#Dropping the other columns
drop= ["food", "recharge", "support", "reminders", "nearby", "movies", "casual", "other", "travel"]
df_test=  df_test.drop(drop,1)

# Convert to lower case
all_text=df_test['message'].str.lower()

# Transform
X_test=tfidf.transform(all_text).toarray()

# Transform Label
y_test=le.transform(df_test['category'])

print(X_test.shape,y_test.shape)

# Predict the values using the above fitted model for X_test
y_pred=log_reg.predict(X_test)

# Calculate the accuracy score
log_accuracy_2=accuracy_score(y_test,y_pred)
print('Logistic Regression Test Accuracy Score',log_accuracy_2)

# Predict the values using the above fitted model for X_test
y_pred=nb.predict(X_test)

# Calculate the accuracy score
nb_accuracy_2=accuracy_score(y_test,y_pred)
print('MultinomialNB Test Accuracy Score',nb_accuracy_2)

# Predict the values using the above fitted model for X_test
y_pred=lsvm.predict(X_test)

# Calculate the accuracy score
lsvm_accuracy_2=accuracy_score(y_test,y_pred)
print('LinearSVC Test Accuracy Score',lsvm_accuracy_2)



# --------------
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim.models.lsimodel import LsiModel
from gensim import corpora
from pprint import pprint
# import nltk
# nltk.download('wordnet')

# Creating a stopwords list
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
# Function to lemmatize and remove the stopwords
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Creating a list of documents from the complaints column
list_of_docs = df["message"].tolist()

# Implementing the function for all the complaints of list_of_docs
doc_clean = [clean(doc).split() for doc in list_of_docs]

# Code starts here

# Create dictionary for all documents
dictionary=corpora.Dictionary(doc_clean)

# Create document word matrix from dictionary for all  documents
doc_term_matrix=[dictionary.doc2bow(doc) for doc in doc_clean]

# Initialise the LSI model
lsimodel=LsiModel(corpus=doc_term_matrix, num_topics=5, id2word=dictionary)

# Print Topics
pprint(lsimodel.print_topics())


# --------------
from gensim.models import LdaModel
from gensim.models import CoherenceModel

# doc_term_matrix - Word matrix created in the last task
# dictionary - Dictionary created in the last task

# Function to calculate coherence values
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    topic_list : No. of topics chosen
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    topic_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(doc_term_matrix, random_state = 0, num_topics=num_topics, id2word = dictionary, iterations=10)
        topic_list.append(num_topics)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return topic_list, coherence_values


# Find Topic List and Coherence Value List
topic_list, coherence_value_list=compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=doc_clean, start=1, limit=41, step=5)

# Find the no. of topics associated with the maximum coherence score
max_index=coherence_value_list.index(max(coherence_value_list))
opt_topic=topic_list[max_index]

# Print 'opt_topic' to take a look at the optimum no. of topics.
print('Optimum No of Topics',opt_topic)

# Initialize LdaModel 
lda_model=LdaModel(corpus=doc_term_matrix, num_topics=opt_topic, id2word = dictionary, iterations=10 , passes=30, random_state=0)

# Print Topics
pprint(lda_model.print_topics(5))


# Code starts here



