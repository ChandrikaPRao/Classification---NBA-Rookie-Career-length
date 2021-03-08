# -*- coding: utf-8 -*-
"""
This program is to solve a classification problem on NBA Rookie Stats dataset 
(provided at data.world). The problem is to predict if a player will last
over 5 years or not.It explores the the following classification/regression 
models, predictc the target value and reports the comparison of their 
performances based F1 scores.
3. Logistic regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2  
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA

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
rsc = RobustScaler()
X_train = rsc.fit_transform(X_train)
X_test = rsc.fit_transform(X_test)

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

# Train the models - Logistic Regression ######################################
#log_reg = LogisticRegression(max_iter=750, penalty='l1')#,solver="lbfgs")#,l1_ratio=0.8)#,penalty='elasticnet',solver="saga")#(max_iter=550, penalty='elasticnet',solver="saga",l1_ratio=1)#solver="lbfgs", max_iter=300
#log_reg = LogisticRegression(max_iter=750, penalty='l2')
#log_reg = LogisticRegression(max_iter=750, penalty='elasticnet',solver="saga",l1_ratio=0.2)
#log_reg = LogisticRegression(max_iter=750, penalty='none',solver="newton-cg")
#log_reg = LogisticRegression(max_iter=750, penalty='none',solver="sag")
#log_reg = LogisticRegression(max_iter=750, penalty='none',solver="saga")

param_grid_l1 = [{'solver': ['liblinear','saga'],'penalty': ['l1']},]
param_grid_l2 = [{'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'penalty': ['l2']},]
param_grid_l2_en = [{'solver': ['saga'],'penalty': ['elasticnet'],'l1_ratio':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]},]
param_grid_l3_n = [{'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],'penalty': ['none']},]
param_grid_list = [param_grid_l1,param_grid_l2,param_grid_l2_en,param_grid_l3_n]
 

logistic_reg = LogisticRegression()
max1 = 0
best_est_logreg_gridsearch = None;
for param in param_grid_list:
    gs_lr = GridSearchCV(logistic_reg, param, cv=10,
                               scoring='f1')
    gs_lr.fit(X_train, y_train)

    #score of each hyperparameter combination tested during the grid search
    result_cv_lr = gs_lr.cv_results_
    for mean_score, params in zip(result_cv_lr["mean_test_score"], result_cv_lr["params"]):
        print(mean_score,params)
        if (mean_score > max1):
            max1 = mean_score
            best_est_logreg_gridsearch = gs_lr.best_estimator_
    final_predictions_lr = gs_lr.best_estimator_.predict(X_test)
print(f1_score(Y_test,final_predictions_lr))


#score of each hyperparameter combination tested during the grid search:
pd.DataFrame(gs_lr.cv_results_)

#Cross Validation and Fine-tune the model
log_reg_cv_results = cross_validate(best_est_logreg_gridsearch, X_train, y_train, cv=10,return_estimator=True,scoring='f1')
max1 = 0;
best_estimator_log_reg = None;
for score, estimator in zip(log_reg_cv_results["test_score"], log_reg_cv_results["estimator"]):
    if (score > max1):
        max1 = score
        best_estimator_log_reg = estimator

print("Best estimator for Logisitic Regression : ", best_estimator_log_reg)
final_predictions = best_estimator_log_reg.predict(X_test)
print("Logistic Regression F1 Score:",round(f1_score(Y_test,final_predictions),5))

