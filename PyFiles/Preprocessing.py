import pandas as pd
import numpy as np
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_transformer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm


def preprocess_headline(headline, pre_type): 
    reg_token = RegexpTokenizer("([a-zA-Z&]+(?:'[a-z]+)?)")

    new_headline = ' '.join([i for i in headline.lower().split() if i != 'rt' and i.endswith('…') == False])
    new_headline  = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",new_headline.lower()).split())
    new_headline  = reg_token.tokenize(new_headline .lower())
    
    if pre_type == 'stem': 
        word_stem = PorterStemmer()
        new_headline = [word_stem.stem(i) for i in new_headline if len(i) > 1]
    elif pre_type == 'lemmet': 
        word_lem = WordNetLemmatizer()
        new_headline= [word_lem.lemmatize(i) for i in new_headline if len(i) > 1]
    
    return ' '.join(new_headline)


def preprocess_steps(preprocessing_dict, x_train, x_test, y_train, y_test, kmeans_cluster = None): 
    #standard scaler volume
    x_train_new = pd.DataFrame()
    x_test_new = pd.DataFrame()
    
    x_train_new['Volume'] = preprocessing_dict['ss_volume'].transform(x_train.Volume.values.reshape(-1,1)).ravel()
    x_test_new['Volume'] = preprocessing_dict['ss_volume'].transform(x_test.Volume.values.reshape(-1,1)).ravel()
    
    #standard scaler daydiff
    x_train_new['DayDiff'] = preprocessing_dict['ss_daydiff'].transform(x_train.DayDiff.values.reshape(-1,1))
    x_test_new['DayDiff'] = preprocessing_dict['ss_daydiff'].transform(x_test.DayDiff.values.reshape(-1,1))
    
    cv_vec = preprocessing_dict['headlines']
    train_headlines = pd.DataFrame(cv_vec.transform(x_train['Headlines']).toarray(), columns = cv_vec.get_feature_names())
    test_headlines = pd.DataFrame(cv_vec.transform(x_test['Headlines']).toarray(), columns = cv_vec.get_feature_names())

    x_train_new = pd.concat([x_train_new, train_headlines], axis = 1)
    x_test_new = pd.concat([x_test_new, test_headlines], axis = 1)
    
    if kmeans_cluster: 
        kmeans = KMeans(n_clusters = kmeans_cluster, max_iter = 1000, tol = 1e-3).fit(x_train_new.values)
        preprocessing_dict['k_cluster'] = kmeans
        x_test_new['KCluster'] = kmeans.predict (x_test_new.values)
        x_train_new['KCluster'] = kmeans.predict(x_train_new.values)

    return x_train_new, x_test_new, y_train, y_test, preprocessing_dict


def get_preprocessing_pickles(x_train, x_test, y_train, y_test, ngram, max_features, min_df, max_df):
    ss_volume = StandardScaler().fit(x_train['Volume'].values.reshape(-1,1))
    ss_daydiff = StandardScaler().fit(x_train['DayDiff'].values.reshape(-1,1))
    cv = CountVectorizer(stop_words = 'english', max_features = max_features, ngram_range = ngram, min_df = min_df,
                                     max_df = max_df).fit(x_train['Headlines'])

    preprocessing_dict = {'ss_volume': ss_volume, 'ss_daydiff': ss_daydiff, 'headlines': cv}
    
    x_train_new, x_test_new, y_train, y_test, preprocessing_dict = preprocess_steps(preprocessing_dict, x_train, x_test, 
                                                                                    y_train, y_test)
    
    return x_train_new, x_test_new, y_train, y_test, preprocessing_dict
    
def preprocess_tts(df, pre_type, ngram, max_features, min_df, max_df): 
    
    pbar = tqdm(total = 3)
    
    pbar.set_description('Preprocessing Headlines')
    df['Headlines'] = df.Headlines.apply(preprocess_headline, pre_type = pre_type)
    pbar.update(1)
    
    X = df[['Headlines', 'Volume', 'DayDiff']]
    Y = df[['Target']]
    
    pbar.set_description('Applying Train Test Split')
    x_train, x_test, y_train, y_test = train_test_split(X,Y, stratify = Y.Target.values, random_state = 10, 
                                                        train_size = .70)
    pbar.update(1)
    
    
    #preprocess the split and return the objects
    pbar.set_description('Getting Preprocessing Objects and Transforming Data')
    x_train_new, x_test_new, y_train, y_test, preprocessing_dict = get_preprocessing_pickles(x_train, 
                                                                                             x_test, y_train, y_test, ngram, 
                                                                                            max_features, min_df, max_df)
    pbar.update(1)
    
    print(f'Train:\t{len(x_train)}\n{y_train.Target.value_counts()}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Test:\t{len(x_test)}\n{y_test.Target.value_counts()}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    y_train = y_train.values.ravel() 
    y_test = y_test.values.ravel()
    return df, x_train_new, x_test_new, y_train, y_test, preprocessing_dict