from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, feature):
        # YOUR CODE HERE
        self.feature = feature

    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        # YOUR CODE HERE
        return self


    def transform(self, X):
        # YOUR CODE HERE
        df = X.copy()
        self.weekday_null_indx = df[df[self.feature].isnull()==True].index
        df.loc[self.weekday_null_indx, self.feature] = df.loc[self.weekday_null_indx, 'dteday'].dt.day_name().apply(lambda x: x[:3])
        return df


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, feature=None):
        self.feature = feature

    def transform(self, X):
        X[self.feature].fillna(self.params_, inplace=True)
        return X

    def fit(self, X,y=None):
        df = X.copy()
        feature_mode = df[self.feature].value_counts().index[0]
        self.params_ = feature_mode
        return self
    
    
class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        #for feature in self.variables:
        #df[self.variables] = df[self.variables].map(self.mappings).astype(int)
        df[self.variables] = df[self.variables].map(self.mappings)
        return df
    
class OutlierHandler(BaseEstimator,TransformerMixin):
    def __init__(self, feature: str):
        self.feature = feature



    def fit(self,X,y=None):
        
        Q1 = np.percentile(X.loc[:, self.feature], 25)
        Q3 = np.percentile(X.loc[:, self.feature], 75)
        deviation_allowed = 1.5*(Q3 - Q1)
        lower_bound = Q1 - deviation_allowed
        upper_bound = Q3 + deviation_allowed

        self.params_ = [lower_bound, upper_bound]

        return self


    def transform(self,X,y=None):
        for i in X.index:
            if X.loc[i,self.feature] > self.params_[1]:
                X.loc[i,self.feature]= self.params_[1]
            if X.loc[i,self.feature] < self.params_[0]:
                X.loc[i,self.feature]= self.params_[0]

        return X
    
    
class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, col):
        # YOUR CODE HERE
        self.col = col

    def fit(self, X: pd.DataFrame, y:pd.Series=None):
        # YOUR CODE HERE
        self.col_list = X[self.col].unique()
        return self

    def transform(self, X):
        # YOUR CODE HERE
        df = X.copy()

        mapping = {}
        for x in range(len(self.col_list)):
          mapping[self.col_list[x]] = x

        one_hot_encode = []
        for c in df[self.col]:
          arr = list(np.zeros(len(self.col_list), dtype = int))
          arr[mapping[c]] = 1
          one_hot_encode.append(arr)

        df[self.col_list] = one_hot_encode
        df.drop(columns = self.col, inplace=True)
        return df
    
class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        df = X.copy()
        df.drop(columns=self.columns,inplace=True, axis=1)
        return df


    def fit(self, X, y=None):
        return self

