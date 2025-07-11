
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
import ast
import lightgbm as lgb
import xgboost as xgb

import requests
from io import BytesIO

# !pip install Pillow
from PIL import Image

# !pip install eli5
import eli5


# !pip install catboost
import catboost

import urllib

from wordcloud import WordCloud
from collections import Counter
from sklearn import feature_extraction
from sklearn import preprocessing
from datetime import datetime
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from keras.metrics import mean_squared_logarithmic_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)



from google.colab import drive
drive.mount('/content/drive/')

root = '/content/drive/My Drive/Course/3rd year/2nd semester/CS464/Project/dataset'
train_path = os.path.join(root, 'train.csv.zip')
test_path = os.path.join(root, 'test.csv.zip')



from google.colab import drive
drive.mount('/gdrive')

root = '/gdrive/My Drive/Bilkent/Year 3'
train_path = os.path.join(root, 'train.csv.zip')
test_path = os.path.join(root, 'test.csv.zip')



train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

print(test.shape)
print(train.shape)

train.info()

missing=train.isna().sum().sort_values(ascending=False)
sns.barplot(missing[:8],missing[:8].index)
plt.style.use('dark_background')
plt.show()

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew',]

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df

dfx = text_to_dict(train)
for col in dict_columns:
       train[col]=dfx[col]

train['belongs_to_collection'].apply(lambda x:len(x) if x!= {} else 0).value_counts()

collections=train['belongs_to_collection'].apply(lambda x : x[0]['name'] if x!= {} else '?').value_counts()[1:15]
sns.barplot(collections,collections.index)
plt.show()

# train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
# train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

# train = train.drop(['belongs_to_collection'], axis=1)

train['tagline'].apply(lambda x:1 if x is not np.nan else 0).value_counts()

plt.figure(figsize=(10,10))
taglines=' '.join(train['tagline'].apply(lambda x:x if x is not np.nan else ''))

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(taglines)
plt.imshow(wordcloud)
plt.title('Taglines')
plt.axis("off")
plt.show()

list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
plt.figure(figsize = (16, 12))
text = ' '.join(['_'.join(i.split(' ')) for j in list_of_keywords for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top keywords')
plt.axis("off")
plt.show()

x=train['production_companies'].apply(lambda x : [x[i]['name'] for i in range(len(x))] if x != {} else []).values
count=Counter([i for j in x for i in j]).most_common(20)
sns.barplot([val[1] for val in count],[val[0] for val in count])

countries=train['production_countries'].apply(lambda x: [i['name'] for i in x] if x!={} else []).values
count=Counter([j for i in countries for j in i]).most_common(10)
sns.barplot([val[1] for val in count],[val[0] for val in count])

train['spoken_languages'].apply(lambda x:len(x) if x !={} else 0).value_counts()

lang=train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
count=Counter([i for j in lang for i in j]).most_common(6)
sns.barplot([val[1] for val in count],[val[0] for val in count])

genre=train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
count=Counter([i for j in genre for i in j]).most_common(10)
sns.barplot([val[1] for val in count],[val[0] for val in count])

dfx = text_to_dict(test)
for col in dict_columns:
  test[col]=dfx[col]

train['log_revenue']=np.log1p(train['revenue'])

train['revenue'].describe()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.scatterplot(train['budget'],train['revenue'])
plt.subplot(1,2,2)
sns.scatterplot(np.log1p(train['budget']),np.log1p(train['revenue']))
plt.show()

train['log_budget']=np.log1p(train['budget'])

plt.hist(train['popularity'],bins=30,color='red')
plt.show()

sns.scatterplot(train['popularity'],train['revenue'],color='green')
plt.show()

"""## Splitting date, into day, month and year"""

def date(x):
    x=str(x)
    year=x.split('/')[2]
    if int(year)<19:
        return x[:-2]+'20'+year
    else:
        return x[:-2]+'19'+year

train['release_date']=train['release_date'].fillna('1/1/90').apply(lambda x: date(x))
test['release_date']=test['release_date'].fillna('1/1/90').apply(lambda x: date(x))

train['release_date']=train['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))
test['release_date']=test['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))

train['release_day']=train['release_date'].apply(lambda x:x.weekday())
train['release_month']=train['release_date'].apply(lambda x:x.month)
train['release_year']=train['release_date'].apply(lambda x:x.year)

test['release_day']=test['release_date'].apply(lambda x:x.weekday())
test['release_month']=test['release_date'].apply(lambda x:x.month)
test['release_year']=test['release_date'].apply(lambda x:x.year)

day=train['release_day'].value_counts().sort_index()
sns.barplot(day.index,day)
plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='45')
plt.ylabel('No of releases')

sns.catplot(x='release_day',y='revenue',data=train, height=20, aspect=1)
plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='90')
plt.title('Revenue on different days of week of release');
plt.show()

sns.catplot(x='release_day',y='runtime',data=train)
plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='90')
plt.show()

plt.figure(figsize=(10,15))
sns.catplot(x='release_month',y='revenue',data=train, height=15, aspect=1)
month_lst = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
plt.gca().set_xticklabels(month_lst,rotation='90')
plt.show()

plt.figure(figsize=(15,8))
yearly=train.groupby(train['release_year'])['revenue'].agg('mean')
plt.plot(yearly.index,yearly)
plt.xlabel('year')
plt.ylabel("Revenue")
plt.savefig('fig')

plt.figure(figsize=(20, 6))
plt.subplot(1, 3, 1)
plt.hist(train['runtime'].fillna(0) / 60, bins=40);
plt.title('Distribution of length of film in hours');
plt.subplot(1, 3, 2)
plt.scatter(train['runtime'].fillna(0), train['revenue'])
plt.title('runtime vs revenue');
plt.subplot(1, 3, 3)
plt.scatter(train['runtime'].fillna(0), train['popularity'])
plt.title('runtime vs popularity');

train['homepage'].value_counts().sort_values(ascending=False)[:5]

genres=train.loc[train['genres'].str.len()==1][['genres','revenue','budget','popularity','runtime']].reset_index(drop=True)
genres['genres']=genres.genres.apply(lambda x :x[0]['name'])

genres=genres.groupby(genres.genres).agg('mean')
plt.figure(figsize=(30,20))
plt.subplot(2,2,1)
sns.barplot(genres['revenue'],genres.index)

plt.subplot(2,2,2)
sns.barplot(genres['budget'],genres.index)

plt.subplot(2,2,3)
sns.barplot(genres['popularity'],genres.index)

plt.subplot(2,2,4)
sns.barplot(genres['runtime'],genres.index)

crew=train['crew'].apply(lambda x:[i['name'] for i in x] if x != {} else [])
Counter([i for j in crew for i in j]).most_common(15)

cast=train['cast'].apply(lambda x:[i['name'] for i in x] if x != {} else [])
Counter([i for j in cast for i in j]).most_common(15)

def prepare_data(df):
  df['_budget_runtime_ratio'] = (df['budget']/df['runtime']).replace([np.inf,-np.inf,np.nan],0)
  df['_budget_popularity_ratio'] = df['budget']/df['popularity']
  df['_budget_year_ratio'] = df['budget'].fillna(0)/(df['release_year']*df['release_year'])
  df['_releaseYear_popularity_ratio'] = df['release_year']/df['popularity']
  df['_releaseYear_popularity_ratio2'] = df['popularity']/df['release_year']
  df['budget']=np.log1p(df['budget'])

  df['collection_name']=df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
  df['has_homepage']=0
  df.loc[(pd.isnull(df['homepage'])),'has_homepage']=1

  le=LabelEncoder()
  le.fit(list(df['collection_name'].fillna('')))
  df['collection_name']=le.transform(df['collection_name'].fillna('').astype(str))

  le=LabelEncoder()
  le.fit(list(df['original_language'].fillna('')))
  df['original_language']=le.transform(df['original_language'].fillna('').astype(str))

  df['_num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
  df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)

  df['isbelongto_coll']=0
  df.loc[pd.isna(df['belongs_to_collection']),'isbelongto_coll']=1

  df['isTaglineNA'] = 0
  df.loc[df['tagline'] == 0 ,"isTaglineNA"] = 1

  df['isOriginalLanguageEng'] = 0
  df.loc[ df['original_language'].astype(str) == "en" ,"isOriginalLanguageEng"] = 1

  df['ismovie_released']=1
  df.loc[(df['status']!='Released'),'ismovie_released']=0

  df['no_spoken_languages']=df['spoken_languages'].apply(lambda x: len(x))
  df['original_title_letter_count'] = df['original_title'].str.len()
  df['original_title_word_count'] = df['original_title'].str.split().str.len()


  df['title_word_count'] = df['title'].str.split().str.len()
  df['overview_word_count'] = df['overview'].str.split().str.len()
  df['tagline_word_count'] = df['tagline'].str.split().str.len()


  df['collection_id'] = df['belongs_to_collection'].apply(lambda x : np.nan if len(x)==0 else x[0]['id'])
  df['production_countries_count'] = df['production_countries'].apply(lambda x : len(x))
  df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))
  df['cast_count'] = df['cast'].apply(lambda x : len(x))
  df['crew_count'] = df['crew'].apply(lambda x : len(x))

  df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
  df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
  df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

  for col in  ['genres', 'production_countries', 'spoken_languages', 'production_companies'] :
      df[col] = df[col].map(lambda x: sorted(list(set([n if n in train_dict[col] else col+'_etc' for n in [d['name'] for d in x]])))).map(lambda x: ','.join(map(str, x)))
      temp = df[col].str.get_dummies(sep=',')
      df = pd.concat([df, temp], axis=1, sort=False)
  df.drop(['genres_etc'], axis = 1, inplace = True)

  cols_to_normalize=['runtime','popularity','budget','_budget_runtime_ratio','_budget_year_ratio','_budget_popularity_ratio','_releaseYear_popularity_ratio',
  '_releaseYear_popularity_ratio2','_num_Keywords','_num_cast','no_spoken_languages','original_title_letter_count','original_title_word_count',
  'title_word_count','overview_word_count','tagline_word_count','production_countries_count','production_companies_count','cast_count','crew_count',
  'genders_0_crew','genders_1_crew','genders_2_crew']
  for col in cols_to_normalize:
      print(col)
      x_array=[]
      x_array=np.array(df[col].fillna(0))
      X_norm=normalize([x_array])[0]
      df[col]=X_norm

  df = df.drop(['belongs_to_collection','genres','homepage','imdb_id','overview','id'
  ,'poster_path','production_companies','production_countries','release_date','spoken_languages'
  ,'status','title','Keywords','cast','crew','original_language','original_title','tagline', 'collection_id'
  ],axis=1)

  df.fillna(value=0.0, inplace = True)

  return df

def get_json(df):
    global dict_columns
    result=dict()
    for col in dict_columns:
        d=dict()
        rows=df[col].values
        for row in rows:
            if row is None: continue
            for i in row:
                if i['name'] not in d:
                    d[i['name']]=0
                else:
                    d[i['name']]+=1
            result[col]=d
    return result




train_dict=get_json(train)
test_dict=get_json(test)

for col in dict_columns :

    remove = []
    train_id = set(list(train_dict[col].keys()))
    test_id = set(list(test_dict[col].keys()))

    remove += list(train_id - test_id) + list(test_id - train_id)
    for i in train_id.union(test_id) - set(remove) :
        if train_dict[col][i] < 10 or i == '' :
            remove += [i]
    for i in remove :
        if i in train_dict[col] :
            del train_dict[col][i]
        if i in test_dict[col] :
            del test_dict[col][i]

test['revenue']=np.nan
all_data=prepare_data((pd.concat([train,test]))).reset_index(drop=True)
train=all_data.loc[:train.shape[0]-1,:]
test=all_data.loc[train.shape[0]:,:]

print(train.shape)

train.drop('revenue',axis=1,inplace=True)
all_data.head()

y=train['log_revenue']
X=train.drop(['log_revenue'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
kfold=KFold(n_splits=3,random_state=42,shuffle=True)

print(X.columns)
print(y)

"""# All metrics used to measure success of an algorithm"""

def show_metrics(y_test, y_pred):
  print("Mean Squared Log Error = " + str(metrics.mean_squared_log_error(y_test, y_pred)))
  print("Root Mean Squared Log Error = " + str(np.sqrt(metrics.mean_squared_log_error(y_test, y_pred))))
  print("Mean Squared Error = " + str(metrics.mean_squared_error(y_test, y_pred)))
  print("Root Mean Squared Error = " + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
  print("R^2 = " + str(metrics.r2_score(y_test, y_pred)))

"""# Artificial neural network"""

from keras import optimizers

model=models.Sequential()
model.add(layers.Dense(356,activation='relu', kernel_regularizer=regularizers.l1(.001), input_shape=(X.shape[1],)))
model.add(layers.Dense(356,activation='relu', kernel_regularizer=regularizers.l1(.001), input_shape=(X.shape[1],)))
model.add(layers.Dense(256,kernel_regularizer=regularizers.l1(.001),activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=optimizers.rmsprop(lr=.001),loss='mse',metrics=['mean_squared_logarithmic_error'])

epochs=100
hist=model.fit(X_train,y_train,epochs=epochs,verbose=0)
test_pred = model.predict(X_test)

show_metrics(y_test, test_pred)

"""# Light Gradient Boosting Machine"""

def msle(y_true, y_pred):
    return 'MSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False

params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 5,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}

model_lgb = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

model_lgb.fit(X_train, y_train,
        verbose=1000)


prediction_lgb_test=model_lgb.predict(X_test)

show_metrics(y_test, prediction_lgb_test)

"""# Random Forest"""

rf_base = RandomForestRegressor(random_state=42)
rf_base.fit(X_train, y_train)
y_rf_base_pred = rf_base.predict(X_test)
print("Base Random Forest Regressor:\n")
show_metrics(y_test, y_rf_base_pred)


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print("\nTuned Random Forest Regressor:\n")
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 75, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)

y_rf_pred = rf_random.predict(X_test)
show_metrics(y_test, y_rf_pred)
print("\nWith the following parameters:")
print(rf_random.best_params_)

"""# XGBoost"""

def xgb_model(X_train, y_train, X_test, y_test) :
  params = {'objective': 'reg:linear',
    'eta': 0.01,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 0.7,
    'eval_metric': 'rmse',
    'seed': 2020,
    'silent': True,
  }

  record = dict()

  train_data = xgb.DMatrix(data=X_train, label=y_train)

  model = xgb.train(dtrain=train_data, num_boost_round=20000, params=params)

  test_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)

  show_metrics(y_test, test_pred)

xgb_model(X_train, y_train, X_test, y_test)

xgb_model(X_train, y_train, X_test, y_test)

"""# CatBoost Regressor"""

from catboost import CatBoostRegressor

def cat_model(X_train, y_train, X_test, y_test) :
  model = CatBoostRegressor(iterations=10000,
  learning_rate=0.004,
  depth=5,
  colsample_bylevel=0.8,
  random_seed = 2020,
  bagging_temperature = 0.2,
  metric_period = None,
  )

  model.fit(X_train, y_train,
    verbose=False)

  test_pred = model.predict(X_test)

  show_metrics(y_test, test_pred)

cat_model(X_train, y_train, X_test, y_test)

"""# Elastic-Net"""

elastic_net_model = ElasticNet(random_state=42)
elastic_net_model.fit(X_train, y_train)
y_elastic_pred = elastic_net_model.predict(X_test)
show_metrics(y_test, y_elastic_pred)

elastic_net_model.get_params()

elastic_netCV_model = ElasticNetCV(cv=5, random_state=42)
elastic_netCV_model.fit(X_train, y_train)
y_elastic_CV_pred = elastic_netCV_model.predict(X_test)
show_metrics(y_test, y_elastic_CV_pred)

"""# Ridged Regression

"""

ridged_model = Ridge(random_state=42)
ridged_model.fit(X_train, y_train)
y_ridge_pred = ridged_model.predict(X_test)
show_metrics(y_test, y_ridge_pred)

ridgeCV_model = RidgeCV(alphas=[0.0005, 0.001, 0.00125, 0.0015, 0.00175, 0.002])
ridgeCV_model.fit(X_train, y_train)
y_ridgeCV_pred = ridgeCV_model.predict(X_test)
show_metrics(y_test, y_ridgeCV_pred)
print("Optimal alpha: " + str(ridgeCV_model.alpha_))

#linear kernel
kernel_ridge_model = KernelRidge(alpha=ridgeCV_model.alpha_)
kernel_ridge_model.fit(X_train, y_train)
y_kernel_ridge_pred = kernel_ridge_model.predict(X_test)
show_metrics(y_test, y_kernel_ridge_pred)

"""# Linear Regression"""

lm = LinearRegression()
lm.fit(X_train, y_train)
y_lm_pred = lm.predict(X_test)
show_metrics(y_test, y_lm_pred)

"""# Support Vector Regression"""

from sklearn.svm import SVR

Cval = 100
eps= 1

SVRreg = SVR(C=Cval, epsilon=eps)
SVRreg.fit(X_train,y_train)
y_pred = SVRreg.predict(X_test)
show_metrics(y_test, y_pred)

"""##Hyper-parameters Fine-Tuning Approaches

#### Cross Validating Params

Note: I cross-validated kernel as well with two possible values; rbf and linear. CV with linear kernel took a long time (+4h), mainly because training with linear kernal is very time consuming. I removed it from here in case the code is re-run. But its results are logged. When all features are used, RBF Kernel is better.
"""

# Cross validation
from sklearn.model_selection import GridSearchCV
params_selection = {
    "C": np.linspace(10, 200, 10),
    "epsilon": np.linspace(.02, 6, 10),
}

clf = GridSearchCV(SVR(), params_selection, verbose=5)
res = clf.fit(X_train, y_train)
# cv_results = cross_validate(SVR(), X_train, y_train, cv=5, fit_params=params_selection, scoring='neg_mean_squared_error', return_estimator=True)

res.cv_results_

"""#### Testing with The Best Selected Modal through Grid Search Cross Validation"""

Best Model Test Results
print("Best Selected Model with params {}".format(res.best_params_))
best_model = res.best_estimator_
y_pred = best_model.predict(X_test)
show_metrics(y_test, y_pred)

"""#### Top 20 Correlated Features

##### Selecting Top 20 Correlated Features
"""

# Select top 20 featues and train and test on that
cor = train.corr(method ='pearson')
cor_target = abs(cor["log_revenue"])
relevant_features = cor_target[cor_target>0.19]
features_names = [f for f in relevant_features.index.values if f != 'log_revenue']
print(features_names)
print(len(features_names))

minimized_X_train = X_train[features_names]
minimized_X_test = X_test[features_names]

"""##### Training with Top 20 Correlated Features using RBF Kernel"""

# when using kernel=rbf
print("Training with Top 20 Correlated Features using RBF Kernel")
SVRreg = SVR(C=200, epsilon=1.67)
SVRreg.fit(minimized_X_train,y_train)
y_pred = SVRreg.predict(minimized_X_test)
show_metrics(y_test, y_pred)

"""##### Training with Top 20 Correlated Features using Linear Kernel"""

print("Training with Top 20 Correlated Features using Linear Kernel")
SVRreg = SVR(C=Cval, epsilon=eps, kernel='linear')
SVRreg.fit(minimized_X_train,y_train)
y_pred = SVRreg.predict(minimized_X_test)
show_metrics(y_test, y_pred)

"""#### Applying PCA"""

from sklearn.decomposition import PCA

pca = PCA(n_components=20)
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)
pca.explained_variance_ratio_

"""##### Training with 20-component PCA Features using RBF Kernel"""

print("Training with 20-component PCA Features using RBF Kernel")
SVRreg = SVR(C=Cval, epsilon=eps)
SVRreg.fit(pca_X_train,y_train)
y_pred = SVRreg.predict(pca_X_test)
show_metrics(y_test, y_pred)

"""##### Training with 20-component PCA Features using Linear Kernel"""

print("Training with 20-component PCA Features using Linear Kernel")
SVRreg = SVR(C=Cval, epsilon=eps, kernel='linear')
SVRreg.fit(pca_X_train,y_train)
y_pred = SVRreg.predict(pca_X_test)
show_metrics(y_test, y_pred)