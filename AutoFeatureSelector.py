#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler

def embedded_lgbm_selector(X, y, num_feats):
    lgbmc = LGBMClassifier(n_estimators=500,
                           learning_rate=0.05,
                           num_leaves=32,
                           colsample_bytree=0.2,
                           reg_alpha=3,
                           reg_lambda=1,
                           min_split_gain=0.01,
                           min_child_weight=40
                          )
    embedded_lgbm_selector = SelectFromModel(lgbmc,
                                             max_features=num_feats
                                            )
    embedded_lgbm_selector = embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature

def embedded_rf_selector(X, y, num_feats):
    rf = RandomForestClassifier(n_estimators=100)
    embedded_rf_selector = SelectFromModel(rf, 
                                           max_features=num_feats
                                          )
    embedded_rf_selector = embedded_rf_selector.fit(X, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    return embedded_rf_support, embedded_rf_feature

def embedded_log_reg_selector(X, y, num_feats):
    logreg = LogisticRegression(penalty='l1', solver='liblinear')
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=50000), max_features=num_feats)
    embedded_lr_selector = embedded_lr_selector.fit(X, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    return embedded_lr_support, embedded_lr_feature

def rfe_selector(X, y, num_feats):
    rf = RandomForestClassifier()
    rfe = RFE(estimator=rf, 
              n_features_to_select=num_feats,
              step=1,
              verbose=5
             )
    rfe = rfe.fit(X, y)
    rfe_support = rfe.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    return rfe_support, rfe_feature

def chi_squared_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    return chi_support, chi_feature

def cor_selector(X, y, num_feats):
    correlations = X.corrwith(y).abs()
    cor_feature = correlations.nlargest(num_feats).index.tolist()
    cor_support = [True if feature in cor_feature else False for feature in X.columns]
    return cor_support, cor_feature

def preprocess_dataset(dataset_path):
    # Load the dataset
    df = pd.read_csv(dataset_path)
    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance', 'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']
    features = numcols + catcols
    df = df[features]
    df = pd.concat([df[numcols], pd.get_dummies(df[catcols])], axis=1)
    df = df.dropna()
    y = df['Overall'] >= 87
    X = df.drop(columns=['Overall'])

    return X, y

def autoFeatureSelector(X, y, method, num_feats):
    if method == 'chi2':
        support, selected_features = chi_squared_selector(X, y, num_feats)
    elif method == 'rfe':
        support, selected_features = rfe_selector(X, y, num_feats)
    elif method == 'logreg':
        support, selected_features = embedded_log_reg_selector(X, y, num_feats)
    elif method == 'lgbm':
        support, selected_features = embedded_lgbm_selector(X, y, num_feats)
    elif method == 'pearson':
        support, selected_features = cor_selector(X, y,num_feats)
    elif method == 'rf':
        support, selected_features = embedded_rf_selector(X, y,num_feats)
    else:
        raise ValueError("Invalid feature selection method entered.")

    return selected_features

if __name__ == "__main__":
    dataset_path = input("Enter your dataset file path here: ")
    available_methods = ['chi2', 'rfe', 'logreg', 'lgbm']
    selected_methods = input(f"Enter feature selection methods from {', '.join(available_methods)} (comma-separated): ").split(',')

    X, y = preprocess_dataset(dataset_path)
    num_feats = 30

    selected_features = set()

    for method in selected_methods:
        selected_features.update(autoFeatureSelector(X, y, method, num_feats))

    selected_features = list(selected_features)

    print(f"Selected features: {', '.join(selected_features)}")


# In[ ]:





# In[ ]:




