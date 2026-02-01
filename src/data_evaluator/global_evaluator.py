import numpy as np
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from .base_evaluator import BaseEvaluator

class GlobalEvaluator(BaseEvaluator) :

    def _get_concatenate_data(self):

        backup_real = self._real.copy(deep=True)
        backup_synth = self._synth.copy(deep=True)

        backup_real['Label'] = np.zeros(self._real.shape[0]).astype('int8')
        backup_synth['Label'] = np.ones(self._synth.shape[0])

        #mix real and synthetic records

        frames = [backup_real, backup_synth]

        #return the concatenate dataframe with the mixed samples
        data = pd.concat(frames).sample(frac=1)

        last_ix = len(data.columns) - 1
        return data.drop(['Label'], axis=1), data[['Label']]


    def model_identification_accuracy(self) :

        X, y = self._get_concatenate_data()

        # define the data preparation for the columns
        t = [('cat', OneHotEncoder(), self._categorical_columns), ('num', MinMaxScaler(), self._numerical_columns)]
        col_transform = ColumnTransformer(transformers=t)
        models = []
        models.append(('LR', LogisticRegression(solver='liblinear')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))

        # evaluate each model in turn
        results = []
        names = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = KFold(n_splits=10, random_state=7, shuffle=True)
            pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])
            cv_results = cross_val_score(pipeline, X, y.values.ravel(), cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        
        return results, names

    def propensity_score(self) :

        X, y = self._get_concatenate_data()

        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)

        t = [('cat', OneHotEncoder(), self._categorical_columns), ('num', MinMaxScaler(), self._numerical_columns)]
        col_transform = ColumnTransformer(transformers=t)

        pipeline = Pipeline(steps=[('prep',col_transform), ('m', LogisticRegression(solver='liblinear'))])
        pipeline.fit(X_train, Y_train)
        proba = pipeline.predict_proba(X_validation)
        return ((proba - 0.5)**2).mean(axis=0), proba




