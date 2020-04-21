### Project Overview

 Haptik is one of the world's largest conversational AI platforms. It is a personal assistant mobile app, powered by a combination of artificial intelligence and human assistance. It has its domain in multiple fields including customer support, feedback, order status and live chat.

We have with us the dataset of Haptik containing the messages it receives from the customers and which topic(class) the messages refer to.

In this project we created a model predicting which class a particular message belongs to using NLP. We also tried to use techniques like LSA (Latent Semantic Analysis) and LDA (Latent Dirichlet Allocation) to assign topics to new messages.

The dataset consists of message column along with the different column associated with the topic they could associated with it.

We had with us two variations of the same dataset

Train data(40000 rows) [We trained our model on this]

Test data(10000 rows) [We validated our model on this]


### Learnings from the project

 oing this project helped us to apply the following skills:

Text preprocessing techniques like Tokenization, Vectorization, etc.

Implementation of Logistic Regression, Naive Bayes and Linear SVM.

Topic modelling with LSA (Latent Semantic Analysis) and LDA (Latent Dirichlet Allocation).

Usage of Coherence Score to determine the optimum number of topics.


### Approach taken to solve the problem

 Our approach to solve this Project was as follows:

Data Cleaning - We loaded the dataset and performed a basic cleaning in order to simplify our futher steps.

Data Processing - We employed a normal TF-IDF vectorizer to vectorize the message column and label encode the category column, essentially making it a classification problem.

Classification Implementation - We applied Logistic Regression , Naive Bayes and Linear SVM model onto the data.

Validation of test data - We saw how well our models runs on test set.

LSI Modeling - We tried to attempt topic modeling on our dataset using Latent Semantic Analysis or LSI.

LDA Modeling - We tried to do topic modeling using Latent Dirichlet Allocation or LDA. We first found the optimum no. of topics using coherence score and then created a model attaining to the optimum no. of topics.


