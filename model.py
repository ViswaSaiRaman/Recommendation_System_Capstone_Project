# Import the necessary libraries
import pandas as pd
import numpy as np
from numpy import * 
import csv
import re 
import string
# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Avoid warnings
import warnings

from sympy import evaluate
warnings.filterwarnings("ignore")
# NLTK libraries
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# Modelling
import pickle
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import xgboost as xgb 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

# To show all the columns
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 300)

# Functions to perform operations. 

#  Function used to fit model and print the metrics
def model_fit(X_train, Y_train, X_test, Y_test, model_obj, print_metrics):
    
    model = model_obj.fit(X_train, Y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = model.score(X_train, Y_train)
    test_acc = model.score(X_test, Y_test)
    perf = classification_report(Y_test, test_pred)
    predict_prob = model.predict_proba(X_test)
    roc_auc = roc_auc_score(Y_test, predict_prob[:,1])
    
    print("Train Accuracy ", train_acc)
    print("Test Accuracy: ", test_acc)
    print(perf)
    print("ROC_AUC score: ", roc_auc)
    print("*************************************************")
    
    if print_metrics == True:
        featureNames = tfidf_vect.get_feature_names()
        coef = model.coef_.tolist()[0]
        coeff_df = pd.DataFrame({'Word' : featureNames, 'Coefficient' : coef})
        coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
        print("************ Top 10 positive features (variables) ************")
        print(coeff_df.head(20).to_string(index=False))
        print("************ Top 10 negative features (variables) ************")        
        print(coeff_df.tail(20).to_string(index=False))
    
    print("Confusion matrix for train and test set")

    plt.figure(figsize=(10,5))

    c_train = confusion_matrix(Y_train, train_pred)
    c_test = confusion_matrix(Y_test, test_pred)
    
    plt.subplot(1,2,1)
    sns.heatmap(c_train/np.sum(c_train), annot=True , fmt = ' .2%', cmap="Greens")

    plt.subplot(1,2,2)
    sns.heatmap(c_test/np.sum(c_test), annot=True , fmt = ' .2%', cmap="Greens")

    plt.show()
    
    # Calculate Sensitivity and Specificity
    true_neg = c_test[0, 0]
    false_pos = c_test[0, 1]
    false_neg = c_test[1, 0]
    true_pos = c_test[1, 1]
    
    sensitivity = true_pos/(false_neg+true_pos)
    print("Sensitivity is : ",sensitivity)
    specificity = true_neg/(false_pos+true_neg)
    print("Specificity is : ",specificity)
    return model

def recommendation(train_df, test_df, dummy_train):
    # Create a pivot table for the user-item matrix
    train_df_pt = train_df.pivot_table(index='username', columns='name', values='rating')

    # Normalising the rating of each user around 0 mean 
    mean = np.nanmean(train_df_pt, axis=1)
    print(mean)
    print(sum(mean))
    # subtract the mean from the train df and transpose
    train_df_sub = (train_df_pt.T - mean).T

    # finding cosine similarity with NaN
    user_correlation_sub = 1 - pairwise_distances(train_df_sub.fillna(0), metric='cosine')
    user_correlation_sub[np.isnan(user_correlation_sub)] = 0

    # User-User prediction 
    # Remove negative correlations among the users
    user_correlation_sub[user_correlation_sub<0]=0
    user_pred_rating = np.dot(user_correlation_sub, train_df_pt.fillna(0))
    user_rating = np.multiply(user_pred_rating,dummy_train)

    return user_rating, user_correlation_sub, train_df_sub

def evaluate(train_df, test_df, dummy_train, user_rating, user_correlation_sub, train_df_sub):
    # Get the common users in both train_df and test_df
    common = test_df[test_df['username'].isin(train_df['username'])]
    print(common.shape)

    # Convert the common users df to pivot table 
    common_pt = common.pivot_table(index='username', columns='name', values='rating')
    # Convert the user_correlation_sub to a dataframe for easy access
    user_corr_df = pd.DataFrame(user_correlation_sub)

    # Change the index of the user_corr_df to usernames
    user_corr_df['username'] = train_df_sub.index
    user_corr_df.set_index('username',inplace=True)

    # Convert the usernames to list
    usernames = common['username'].tolist()
    user_corr_df.columns = train_df_sub.index.tolist()
    user_corr_df_1 =  user_corr_df[user_corr_df.index.isin(usernames)]
    user_corr_df_2 = user_corr_df_1.T[user_corr_df_1.T.index.isin(usernames)]
    user_corr_df_3 = user_corr_df_2.T
    # Mark the negative correlations to 0 
    user_corr_df_3[user_corr_df_3<0]=0

    # predict the user rating 
    common_pred_rating = np.dot(user_corr_df_3, common_pt.fillna(0))

    # Create a dummy test dataframe 
    test_dummy = common.copy()
    # make the rating to 1 if the user has given a rating, else 0
    test_dummy['rating'] = test_dummy['rating'].apply(lambda x : 1 if x>=1 else 0)
    # Create a pivot table for the dummy test dataframe
    test_dummy_pt = test_dummy.pivot_table(index='username', columns='name', values='rating').fillna(0)

    # predict the final testing 
    common_pred_rating = np.multiply(common_pred_rating, test_dummy_pt)
    
    # normalize the range of the rating
    x  = common_pred_rating.copy() 
    x = x[x>0]
    scaler = MinMaxScaler(feature_range=(1, 5))
    scaler.fit(x)
    y = (scaler.transform(x))

    # Creating a pivot table of common users
    common_1 = common.pivot_table(index='username', columns='name', values='rating')
    total_non_nan = np.count_nonzero(~np.isnan(y))
    # calculate rmse 
    rmse = (sum(sum((common_1 - y )**2))/total_non_nan)**0.5
    print(rmse)

# Recommend top 5 items for a user
def recommend_top(user_name):
    # top20 ratings for a particular user
    top20 = user_rating.loc[user_name].sort_values(ascending=False)[0:20]
    # using lr predict the sentiment score of a product based on the review.

    items = []
    label = []
    for item in top20.index.tolist():
        # get the item name
        item_review = reviews[reviews['name']==item]['text'].tolist()
    
        # get the features of the review
        tfidf_features = tfidf_vect.transform(item_review)
        items.append(item)
        label.append(lr.predict(tfidf_features).mean())
    
    # Create a dataframe of the items and the label
    items_df = {'Item' : items, 'Label' : label}
    items_df = pd.DataFrame(items_df)
    # Sort the dataframe based on the sentiment score in the descending order
    items_df = items_df.sort_values(by=['Label'], ascending=False)

     # top 5 recommendations will be the following
    top5 = items_df['Item'][0:5].tolist()
    return top5

# Pre process the data
def preprocess(df):
    # Drop the columns which have high missing values
    df = df.drop(columns=['reviews_userCity', 'reviews_userProvince'])
    # Drop the columns which are not useful
    df = df.drop(columns=['reviews_didPurchase', 'reviews_doRecommend'])

    # Drop all those rows where manufacturer, reviews_date, reviews_title
    # reviews_username, user_sentiment  is null
    df = df[df['manufacturer'].notna()]
    df = df[df['user_sentiment'].notna()]
    df = df[df['reviews_title'].notna()]
    df = df[df['reviews_date'].notna()]
    df = df[df['reviews_username'].notna()]
    
    # Remove the prefix word reviews_ from the column names
    df.columns = list(map((lambda x : x.lstrip("reviews_") if x.startswith("reviews_") else x), list(df.columns)))
    df = df.rename(columns={'ating':'rating'})
    
    # convert target variable to a binary value for modeling 
    df['user_sentiment'] = df['user_sentiment'].apply(lambda x : 1 if x=="Positive" else 0)

    # Convert the username, title and text to lowercase and strip
    df['username'] = df['username'].apply(lambda x : x.lower())
    df['username'] = df['username'].apply(lambda x : x.strip())
    df['title'] = df['title'].apply(lambda x : x.lower())
    df['title'] = df['title'].apply(lambda x : x.strip())
    df['text'] = df['text'].apply(lambda x : x.lower())
    df['text'] = df['text'].apply(lambda x : x.strip()) 
    return df

# function to remove stop words
def delete_stopwords(text):
    stopwords_obj = stopwords.words('english')
    word = [i for i in text.split() if i not in stopwords_obj]
    return ' '.join(word)

# function to lemmatize the sentence
def lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    sent = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]
    return " ".join(sent)

# Tokenization and Lemmatization using two functions above
def text_processing(df):
    df['review'] = df['text']
    df['review'] = df['review'].str.strip()
    # Delete all the punctuation marks
    df['review'] = df['review'].str.replace('[^\w\s]','')
    # Delete stop words 
    df['review'] = df['review'].apply(delete_stopwords)
    # Lemmatize every word in a sentence
    df['review'] = df['review'].apply(lemmatizer)    
    return df

# Visualization functions
def visualize_1(df):
    # Visualize the doc length against the number of reviews
    plt.figure(figsize=(6,6))
    lens = [len(i) for i in df['review']]
    plt.hist(lens, bins = 10)
    plt.xlabel('Review Length')
    plt.ylabel('Number of Reviews')
    plt.show()
def visualize_2(df):
    # Visualize no: of positive and negative reviews
    plt.hist(df['user_sentiment'])
    plt.xlabel('Class')
    plt.ylabel('Number of Reviews')
    plt.show()



#------------------------------------------------------------------------------------------------#
# reading the data

reviews = pd.read_csv("sample30.csv")
print(reviews.shape)
print(list(reviews.columns))
print(reviews.head())

# data preprocessing, text processing, visualization
reviews = preprocess(reviews)
reviews = text_processing(reviews)
visualize_1(reviews)
visualize_2(reviews)

# Assign the X and Y variables
X = reviews['review']
Y = reviews['user_sentiment']

# test train split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)

print('X_train', X_train.shape)
print('Y_train', Y_train.shape)
print('X_test', X_test.shape)
print('Y_test', Y_test.shape)

# tfidf vector initialization
tfidf_vect = TfidfVectorizer(ngram_range=(1,3), lowercase=True, analyzer='word',stop_words= 'english',
                        token_pattern=r'\w{1,}')
# fit the model
tfidf_vect.fit(X_train)

## transforming the train and test datasets
X_train = tfidf_vect.transform(X_train.tolist())
X_test = tfidf_vect.transform(X_test.tolist())

# Since we saw huge class imbalance in the previous outputs,  its better we rectify it first before training the model
smote = SMOTE()

# Apply smote on train and test datasets.
X_train, Y_train = smote.fit_resample(X_train, Y_train)
X_test, Y_test = smote.fit_resample(X_test, Y_test)

# Initialize logistic regression object - Best model
lr_obj = LogisticRegression()
lr = model_fit(X_train, Y_train, X_test, Y_test, lr_obj, True)

# Copy the original df to a new dataframe 
reviews_temp = reviews

# Diving the dataset into train and test 
train_df, test_df = train_test_split(reviews_temp, train_size=0.7, test_size=0.3, random_state=100)

print(train_df.shape)
print(test_df.shape)

# Create a temporary pivot table with index as username 
user_pt = reviews.pivot_table(index='username', columns='name', values='rating').fillna(0)
user_df = pd.DataFrame(user_pt)

# Create a dummy_train dataframe - used for predicting unrated items by user
dummy_train = train_df.copy()

# As we know we need to mark the items which have not been rated by the user as 1
dummy_train.rating = dummy_train.rating.apply(lambda x : 0 if x>=1 else 1)
dummy_train = dummy_train.pivot_table(index='username', columns='name', values='rating').fillna(1)

# predict the rating
user_rating, user_correlation_sub, train_df_sub = recommendation(train_df, test_df, dummy_train)
# evaluate the recommendation system 
evaluate(train_df, test_df, dummy_train, user_rating, user_correlation_sub, train_df_sub)

# top 5 recommendations 
user = input("Enter a user name :")
top5 = recommend_top(user)
print(top5)

# Save the final df to a csv
reviews.to_csv("reviews_df.csv",index=False)

# Saving to pickle files
# dump the logistic regression, user_rating and tfidf_vect to pkl files
pickle.dump(lr,open('models/best_model_lr.pkl', 'wb'))
pickle.dump(tfidf_vect,open('models/tfidf_vec.pkl','wb'))
pickle.dump(user_rating, open('models/user_item_ratings.pkl', 'wb'))

# Load pkl objects
lr_pkl =  pickle.load(open('models/best_model_lr.pkl', 'rb'))
tfidf_pkl = pickle.load(open('models/tfidf_vec.pkl','rb'))
ratings_pkl = pickle.load(open('models/user_item_ratings.pkl', 'rb'))




#App is Deployed to Heroku and is available @ https://capstone-raman.herokuapp.com/
#github-link : https://github.com/ViswaSaiRaman/Recommendation_System_Capstone_Project





