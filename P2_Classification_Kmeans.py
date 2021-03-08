# -*- coding: utf-8 -*-
"""
This program is to solve a classification problem on NBA Rookie Stats dataset 
(provided at data.world). The problem is to predict if a player will last
over 5 years or not.It explores the the following classification/regression 
models, predictc the target value and reports the comparison of their 
performances based F1 scores.
1. K-nearest neighbors
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
norm = Normalizer()
X_train_norm = norm.fit_transform(X_train)
X_test_norm = norm.fit_transform(X_test)

#norm = Normalizer()
#X_train_norm = X_train
#X_test_norm = X_test


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

# Train the models - K-nearest neighbors ########################################
    
knnCls = KNeighborsClassifier()
knn_param_grid = [
        {'n_neighbors':[5,10,15,20], 'p':[1,2,3], 
         'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
         'leaf_size':[30,40,50,60]}
        ] 
grid_search_knn = GridSearchCV(knnCls, knn_param_grid, cv = 10, scoring="f1")

grid_search_knn.fit(X_train, y_train)
knn_best_estimator = grid_search_knn.best_estimator_
knn_cv_results = grid_search_knn.cv_results_
print("knn gridSearch best_score for original data =", grid_search_knn.best_score_)

grid_search_knn.fit(X_train_norm, y_train)
knn_best_estimator_norm = grid_search_knn.best_estimator_
knn_cv_results_norm = grid_search_knn.cv_results_
print("knn gridSearch best_score for normalized data =", grid_search_knn.best_score_)

score_result_dict = {}
score_result_dict["equal"] = 0
score_result_dict["nom"] = 0
score_result_dict["orig"] = 0
for mean_score, params in zip(knn_cv_results_norm["mean_test_score"],
knn_cv_results_norm["params"]):
    for mean_score1, params1 in zip(knn_cv_results["mean_test_score"],
knn_cv_results["params"]):
        if(params == params1):
            if(mean_score == mean_score1):
                score_result_dict["equal"] += 1
            elif(mean_score < mean_score1):
                score_result_dict["orig"] += 1
            else:
                score_result_dict["nom"] += 1
#score_result_list
#print("score_result_list :",score_result_list)

norm_final_pred = knn_best_estimator_norm.predict(X_test_norm)
print("F1 Score norm_estimator:",f1_score(Y_test,norm_final_pred))
print(knn_best_estimator_norm)

orig_final_pred = knn_best_estimator.predict(X_test)
print("F1 Score original_estimator:",f1_score(Y_test,orig_final_pred))
print(knn_best_estimator)

cv_results_knn = cross_validate(knn_best_estimator_norm, X_train, y_train, cv=10,return_estimator=True,scoring='f1')
max1 = 0;
for score, estimator in zip(cv_results_knn["test_score"],cv_results_knn["estimator"]):
    if (score > max1):
        max1 = score
        best_estimator_knn = estimator
#print(max1)
print("Best estimator for Kmeans : ", best_estimator_knn)

final_predictions = best_estimator_knn.predict(X_test)
print("F1 Score for K Nearest Neighbors:",round(f1_score(Y_test,final_predictions),5))
















