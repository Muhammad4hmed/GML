import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LinearRegression, LassoLarsCV, BayesianRidge, SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import warnings
warnings.filterwarnings("ignore")

class AutoML:
    def __init__(self):
        self.clf_models = [LogisticRegression(max_iter=10000), 
                           DecisionTreeClassifier(), 
                           KNeighborsClassifier(), SGDClassifier(),  RandomForestClassifier(n_jobs=-1),  
                           AdaBoostClassifier(),  
                           ExtraTreesClassifier(n_jobs=-1),  XGBClassifier(n_jobs=-1),  LGBMClassifier(n_jobs=-1),  
                           CatBoostClassifier(verbose=0),  
                           GradientBoostingClassifier(),  GaussianNB(),  MLPClassifier(max_iter = 1000)  ]
        self.reg_models = [LassoLarsCV(),  LinearRegression(),  DecisionTreeRegressor(),  KNeighborsRegressor(),  
                          SGDRegressor(),  RandomForestRegressor(n_jobs=-1),  AdaBoostRegressor(),  ExtraTreesRegressor(n_jobs=-1),  
                          XGBRegressor(n_jobs=-1),  LGBMRegressor(n_jobs=-1),  CatBoostRegressor(verbose=0),  
                          GradientBoostingRegressor(),  
                          BayesianRidge(),  MLPRegressor(max_iter = 1000) ]
    def plot_table(self, table):
        fig, axs = plt.subplots(1,1, figsize=(15,2))
        collabel=tuple(table.columns)
        axs.axis('tight')
        axs.axis('off')
        the_table = axs.table(cellText=table.values,colLabels=collabel,loc='upper center')
        return fig
    def plot_learning_curve(self,estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), name = None):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        axes : array of 3 axes, optional (default=None)
            Axes to use for plotting the curves.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        print(name)
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return np.mean(test_scores_mean)
    
    def cross_val(self, X, y, model, metric, folds):
        scores = []
        for tr_in, val_in in KFold(n_splits = folds).split(X, y):
            model_fold = model
            X_train, y_train, X_val, y_val = X.iloc[tr_in,:], y[tr_in], X.iloc[val_in,:], y[val_in]
            model_fold.fit(X_train, y_train)
            y_hat = model.predict(X_val)
            scores.append(metric(y_val,y_hat))
        return np.mean(scores)
    
    def GMLClassifier(self, X, y, metric = accuracy_score, folds = 5):
        '''
        X: Independent variable(s)
        y: Dependent variable
        metric: metric of evaluation 
        folds: Number of validation folds, default: 5
        '''
        result = pd.DataFrame(columns = ['Models','Scores'])
        for i,model in enumerate(self.clf_models):
            name = str(model.__class__.__name__)
            scores = self.cross_val(X, y, model, metric, folds)
            
            print('{} got score of {} in {} folds'.format(name,scores,folds))
            result.at[i, 'Models'] = name
            result.at[i, 'Scores'] = scores
        result = result.sort_values('Scores',ascending=False).reset_index(drop=True)
        self.plot_table(result)
    
    def GMLRegressor(self, X, y, metric = mean_absolute_error,folds = 5):
        '''
        X: Independent variable(s)
        y: Dependent variable
        metric: metric of evaluation 
        folds: Number of validation folds, default: 5
        '''
        result = pd.DataFrame(columns = ['Models','Scores'])
        for i,model in enumerate(self.reg_models):
            name = str(model.__class__.__name__)
            scores = self.cross_val(X, y, model, metric, folds)
            
            print('{} got score of {} in {} folds'.format(model.__class__.__name__,scores,folds))
            result.at[i, 'Models'] = name
            result.at[i, 'Scores'] = scores
        result = result.sort_values('Scores').reset_index(drop=True)
        self.plot_table(result)