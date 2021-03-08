# -*- coding: utf-8 -*-
"""
This program is to solve a classification problem on NBA Rookie Stats dataset 
(provided at data.world). The problem is to predict if a player will last
over 5 years or not.It explores the the following classification/regression 
models, predictc the target value and reports the comparison of their 
performances based F1 scores.
2. Random forests 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2  
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate

np.random.seed(42)

def display_scores(scores, stype):
    print(stype, "Scores:", scores)
    print(stype, "Mean:", scores.mean())
    print(stype, "Standard deviation:", scores.std())
    
nba_data = pd.read_csv('nba.csv')
#    print(nba_data.head())
#    print(nba_data.shape)
#    print(nba_data.info())
#    print(nba_data.describe())

# Data pre-processing part 1 ##################################################
    # Getting rid of redundant data
nba_data.drop_duplicates(keep='first',inplace=True)    

    #Correlation graph
#plt.matshow(nba_data.corr())
#plt.show()
    
    #The below correlation matrix clearly shows that the 3P% is correlated marginally to 3PM and 3PA.
#f, ax = plt.subplots(figsize=(10, 8))
#corr = nba_data.corr()
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
#        cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
    # Target dependency plot
#plt.scatter(nba_data['3P%'], nba_data['TAR'])
#plt.show()    

    # Update the blank rows in 3P% 
filtered_df = nba_data.loc[:,['3PM','3PA','3P%']]
filtered_df = filtered_df[filtered_df['3PM'] == 0]
filtered_df = filtered_df[filtered_df['3PA'] == 0]
avg_3P_val = filtered_df['3P%'].mean()
nba_data["3P%"].fillna(avg_3P_val, inplace = True) 

#     Check the importance score of each column
#c_data = nba_data.copy()    
#bestfeatures = SelectKBest(score_func=chi2, k="all")
#fit = bestfeatures.fit(nba_data.drop(columns=["Name","TAR"]),nba_data["TAR"])
#dfscores = pd.DataFrame(fit.scores_)
#dfcolumns = pd.DataFrame(nba_data.drop(columns=["Name","TAR"]).columns)
#featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
#featureScores.columns = ['Specs','Score']
#print(featureScores)
#
#print(featureScores.nlargest(19,'Score')) 

# End of Data preprocessing part 1 -------------------------------------------    
# Split the training and the test set #########################################

train_set, test_set = train_test_split(nba_data, test_size = 0.2, random_state = 42)

y_train = train_set["TAR"]
X_train = train_set.drop(columns=["Name","TAR"])#["Name","FGM","FGA","3PM","3PA","FTM","FTA","TAR"])#["Name","FGM","FGA","3PM","3PA","FTM","FTA","TAR"])#["Name","3PM","3PA","3P%","TAR"]["Name","TAR"]
Y_test = test_set["TAR"]
X_test = test_set.drop(columns=["Name","TAR"])#["Name","FGM","FGA","3PM","3PA","FTM","FTA","TAR"])

# Feature scaling##############################################################
# Standard scaler ###########
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)

# Normalizer ###########
#transformer = Normalizer().fit(X_train)
#transformer.transform(X_train)
#transformer_t = Normalizer().fit(X_test)
#transformer_t.transform(X_test)

# MinMaxscaler ###########
#mmsc = MinMaxScaler()
#X_train = mmsc.fit_transform(X_train)
#X_test = mmsc.fit_transform(X_test)

# Robust scaler ###########
#rsc = RobustScaler()
#X_train = rsc.fit_transform(X_train)
#X_test = rsc.fit_transform(X_test)

# Quantile ##############
#sc = QuantileTransformer()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
###
###
#from sklearn.kernel_approximation import AdditiveChi2Sampler
#sc = AdditiveChi2Sampler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# PCA ########################
#pca = PCA(n_components=2)  
#X_train = pca.fit_transform(X_train)
#X_test = pca.fit_transform(X_test)   
#    
#   

# Train the models - Random Forests ########################################

random_forest_param_grid = [
    {'n_estimators': [30, 60, 100, 120], 'bootstrap': [True,False]},
  ]

random_forest_clas = RandomForestClassifier()
random_forest_clas.get_params().keys()


gs_rand_forest = GridSearchCV(random_forest_clas, random_forest_param_grid, cv=10,
                           scoring='f1')
gs_rand_forest.fit(X_train, y_train)
rand_forest_cv = gs_rand_forest.cv_results_

print(gs_rand_forest.best_score_)
final_predictions_rf = gs_rand_forest.best_estimator_.predict(X_test)
print(f1_score(Y_test,final_predictions_rf))

rf_cv_results = cross_validate(gs_rand_forest.best_estimator_, X_train, y_train, cv=10,return_estimator=True,scoring='f1')
max1 = 0;
best_estimator_rf = None;
for score, estimator in zip(rf_cv_results["test_score"], rf_cv_results["estimator"]):
    if (score > max1):
        max1 = score
        best_estimator_rf = estimator
#print(max1)
print("Best estimator for Random Forest : ", best_estimator_rf)
final_predictions = best_estimator_rf.predict(X_test)
print("F1 Score for Random Forest Classifier:",round(f1_score(Y_test,final_predictions),5))



