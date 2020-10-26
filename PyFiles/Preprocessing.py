import pandas as pd
import numpy as np
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm


# def get_target(x): 
#     if abs(x) >= 0 and abs(x) < 1: 
#         return 1 
#     elif x >= 1: 
#         return 2 
#     elif x <= -1: 
#         return 0
#     else: 
#         return None
    
def get_target(x): 
    if x < 0: 
        return 0 
    elif x >= 0: 
        return 1 
    else: 
        return None

def combine_ticker(ticker_df, grouped_df, path):
    ticker_df['Time'] = pd.to_datetime(ticker_df['Time'], errors = 'coerce').dt.normalize()
    
    #retrieve values that are found in the range of the headlines df
    ticker_seg = ticker_df[(ticker_df.Time >= grouped_df.Time.min()) & (ticker_df.Time <= grouped_df.Time.max())]
    
    #set index equal to each other for merging
    ticker_seg = ticker_seg.set_index(pd.to_datetime(ticker_seg.Time)).drop('Time', axis = 1)
    grouped_df = grouped_df.set_index(pd.to_datetime(grouped_df.Time)).drop('Time', axis = 1)
    
    #merge the two df
    ticker_grouped = pd.concat([grouped_df, ticker_seg], axis = 1, join = 'inner')
    
    #get the difference in closing price
    ticker_grouped['CloseDiff'] = ticker_grouped['4. close'].diff()
    
    close_diff = []
    for diff in ticker_grouped.CloseDiff.values[1:]: 
        close_diff.append(diff)
    close_diff.append(None)
    ticker_grouped['CloseDiffNew'] = close_diff
    
    #get binary target values for going up or down
    ticker_grouped['Target'] = ticker_grouped.CloseDiffNew.map(get_target)
    ticker_grouped.dropna(subset = ['Target', 'CloseDiffNew'],inplace = True)
    
    #get the difference in price on a day
    ticker_grouped['DayDiff'] = ticker_grouped['4. close'] - ticker_grouped['1. open'] 
    
    #rename columns
    ticker_grouped = ticker_grouped.rename(columns = {'5. volume': 'Volume'})
    ticker_grouped[['Headlines', 'Volume', 'DayDiff', 'Target']]
    ticker_grouped.to_csv(f'FData/Headlines/New/{path}')
    
    return ticker_grouped

def group_dates(combined_df): 
    grouped_df = pd.DataFrame()
    unique_dates = combined_df.Time.unique()
    grouped_headlines = []
    pbar = tqdm(unique_dates, desc = 'Grouping Rows By Dates')
    for date in pbar: 
        temp_df = combined_df[combined_df.Time == date]
        headlines = temp_df.Combined.values 
        combined_headlines = ' '.join(headlines)
        grouped_headlines.append(combined_headlines)
     
    
    grouped_df['Headlines'] = grouped_headlines 
    grouped_df['Time'] = unique_dates
    grouped_df = grouped_df.sort_values('Time', ascending = True).reset_index(drop = True)
    grouped_df.dropna(subset = ['Time'], inplace = True)
    pbar.close()
    return grouped_df

def get_ticker_df(ticker): 
    from PyFiles import config
    from alpha_vantage.timeseries import TimeSeries
    
    api_key = config.api_key

    ts = TimeSeries(key = api_key, output_format = 'pandas')

    data_ts, meta_ts = ts.get_daily(symbol = ticker, outputsize = 'full')

    data_ts['Time'] = data_ts.index
    data_ts.to_csv(f'FData/Headlines/New/{ticker}Daily.csv', index = False)
    
    return data_ts

def remove_b(x): 
    if x[:2] == 'b"' or x[:2] == "b'": 
        x = x[1::]
        return x
    else: 
        return x
def combine_headlines_descriptions(df): 
    df = df.dropna(subset = ['Time'])
    headlines = np.array([[i] for i in df.Headlines.values])
    descriptions = np.array([[i] for i in df.Description.values])
    combined = np.concatenate((headlines, descriptions), axis = 1)
    new_combined = []
    for i in combined: 
        new_combined.append(' '.join(i))
    df['Combined'] = new_combined
    df = df.drop(['Headlines', 'Description'], axis = 1)
    return df

def combine_headlines(ticker, path = None):
    cnbc = pd.read_csv('FData/Headlines/cnbc_headlines.csv')
    guardian = pd.read_csv('FData/Headlines/guardian_headlines.csv')
    reuters = pd.read_csv('FData/Headlines/reuters_headlines.csv')
    reddit = pd.read_csv('FData/Headlines/RedditNews.csv')
#     stocker_bot = pd.read_csv('FData/Headlines/stockerbot-export1.csv')

    
    cnbc['Time'] = pd.to_datetime(cnbc['Time'], errors = 'coerce').dt.normalize()
    guardian['Time'] = pd.to_datetime(guardian['Time'], errors = 'coerce').dt.normalize()
    reuters['Time'] = pd.to_datetime(reuters['Time'], errors = 'coerce').dt.normalize()
    
    #combining description with the headlines
    cnbc = combine_headlines_descriptions(cnbc)
    reuters = combine_headlines_descriptions(reuters)
    
    #renaming guardian and reddit headline for grouping
    guardian = guardian.rename(columns = {'Headlines': 'Combined'})
    reddit = reddit.rename(columns = {'Headlines': 'Combined'})
    
    #adding source to each df
    guardian['Source'] = ['Guardian' for i in range(len(guardian))]
    cnbc['Source'] = ['CNBC' for i in range(len(cnbc))]
    reddit['Source'] = ['Reddit' for i in range(len(reddit))]
    reuters['Source'] = ['Reuters' for i in range(len(reuters))]
    
    #decoding the strings in reddit 
    reddit['Combined'] = reddit.Combined.map(remove_b)
    
    #combining the datasets 
    combined_df = pd.concat([guardian, cnbc, reuters, reddit])
    
    df_dict = {'cnbc': cnbc, 'reuters': reuters, 'reddit': reddit, 'guardian': guardian, 'combined': combined_df}
    
    combined_df['Time'] = pd.to_datetime(combined_df['Time'], errors = 'coerce').dt.normalize()
    
    #save combinedHeadlines
    if path:
        combined_df.to_csv(f'FData/Headlines/New/{path}')
    
    #group the headlines by day
    grouped_df = group_dates(combined_df)
    
    #get the ticker df
    ticker_df = get_ticker_df(ticker) 
    
    #join the ticker_df and headline df by day
    ticker_grouped = combine_ticker(ticker_df, grouped_df, path)
    ticker_grouped = ticker_grouped.reset_index()
    return ticker_grouped, df_dict



def preprocess_headline(headline, pre_type = None): 
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
    x_train_new['DayDiff'] = preprocessing_dict['ss_daydiff'].transform(x_train.DayDiff.values.reshape(-1,1)).ravel()
    x_test_new['DayDiff'] = preprocessing_dict['ss_daydiff'].transform(x_test.DayDiff.values.reshape(-1,1)).ravel()
    
    #standardscaler low
    x_train_new['Low'] = preprocessing_dict['ss_low'].transform(x_train['3. low'].values.reshape(-1,1)).ravel()
    x_test_new['Low'] = preprocessing_dict['ss_low'].transform(x_test['3. low'].values.reshape(-1,1)).ravel()
    
    vect = preprocessing_dict['headlines']
    train_headlines = pd.DataFrame(vect.transform(x_train['Headlines']).toarray(), columns = vect.get_feature_names())
    test_headlines = pd.DataFrame(vect.transform(x_test['Headlines']).toarray(), columns = vect.get_feature_names())

    x_train_new = pd.concat([x_train_new, train_headlines], axis = 1)
    x_test_new = pd.concat([x_test_new, test_headlines], axis = 1)
    
    if kmeans_cluster: 
        kmeans = KMeans(n_clusters = kmeans_cluster, max_iter = 1000, tol = 1e-3).fit(x_train_new.values)
        preprocessing_dict['k_cluster'] = kmeans
        x_test_new['KCluster'] = kmeans.predict (x_test_new.values)
        x_train_new['KCluster'] = kmeans.predict(x_train_new.values)

    return x_train_new, x_test_new, y_train, y_test


def get_preprocessing_objects(x_train, ngram, max_features, min_df, max_df, vect_type = 'cv'):
    ss_volume = StandardScaler().fit(x_train['Volume'].values.reshape(-1,1))
    ss_daydiff = StandardScaler().fit(x_train['DayDiff'].values.reshape(-1,1))
    ss_low = StandardScaler().fit(x_train['3. low'].values.reshape(-1,1))
    sw = stopwords.words('english')
    if vect_type == 'cv': 
        vect = CountVectorizer(stop_words = sw, max_features = max_features, ngram_range = ngram, min_df = min_df,
                                     max_df = max_df).fit(x_train['Headlines'])
    elif vect_type == 'tfidf': 
        vect = TfidfVectorizer(stop_words = sw, max_features = max_features, ngram_range = ngram, min_df = min_df,
                                     max_df = max_df).fit(x_train['Headlines'])

    preprocessing_dict = {'ss_volume': ss_volume, 'ss_daydiff': ss_daydiff, 'ss_low': ss_low, 'headlines': vect}
     
    return preprocessing_dict
    
def preprocess_tts(df, pre_type, ngram, max_features, min_df, max_df, vect_type): 
    new_df = df.copy()
    pbar = tqdm(total = 3)
    
    pbar.set_description('Preprocessing Headlines')
    new_df['Headlines'] = new_df.Headlines.apply(preprocess_headline, pre_type = pre_type)
#     return new_df, 1,1,1,1,1
    pbar.update(1)
    
    X = new_df[['Headlines', 'Volume', 'DayDiff', '3. low']]
    Y = new_df[['Target']]
    
    pbar.set_description('Applying Train Test Split')
    x_train, x_test, y_train, y_test = train_test_split(X,Y, stratify = Y.Target.values, random_state = 10, 
                                                        train_size = .8)
    pbar.update(1)
    
    
    #preprocess the split and return the objects
    pbar.set_description('Getting Preprocessing Objects and Transforming Data')
    #getting the transformers
    preprocessing_dict = get_preprocessing_objects(x_train, ngram, max_features, min_df, max_df, vect_type)  
    #applying transformers to data 
    x_train_new, x_test_new, y_train, y_test = preprocess_steps(preprocessing_dict, x_train, x_test, 
                                                                                    y_train, y_test)
    pbar.update(1)
    
    print(f'Train:\t{len(x_train)}\n{y_train.Target.value_counts()}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Test:\t{len(x_test)}\n{y_test.Target.value_counts()}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    y_train = y_train.values.ravel() 
    y_test = y_test.values.ravel()
    
    pbar.close()
    return new_df, x_train_new, x_test_new, y_train, y_test, preprocessing_dict







