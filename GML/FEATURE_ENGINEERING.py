import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import base64
from .AUTO_FEATURE_ENGINEERING import AutoFeatClassifier, AutoFeatRegressor
from sklearn.preprocessing import *
from category_encoders import *

import random
import string

iqr = 0
upper_inner_fence = 0
lower_inner_fence = 0
upper_outer_fence = 0
lower_outer_fence = 0
percentage1 = 0
percentage2 = 0
len_ds = 0
len_out = 0

class FeatureEngineering:
    def __init__(self, data, label, fill_missing_data = True, method_cat = 'Mode', method_num = 'Mean', 
                 drop = True, cat_cols = None, numeric_cols = None, thresh_cat = 0.10, thresh_numeric = 0.10, 
                 encode_data = True, method = 'TargetEncoder', thresh = 0.10,
                 normalize = True, method_transform='LogTransform', thresh_numeric_transform = 0.10, 
                 remove_outliers = True, qu_fence = 'inner', new_features = True, task = 'classification', 
                 test_data = None, verbose=1, feateng_steps=2):
        """
        Call get_new_data() to get:
            X,y and test_data(if any)
        
        data: data 
        label: name of label/target column
        
        fill_missing_data: Should it fill missing data? 
        method_cat: possible methods are: Mode(default), Constant and Interpolate
        method_num: possible methods are: Mean(default), Median, Mode, Constant and Interpolate
        drop: if the missing ratio is greater than available data, should the feature be dropped(default) or not
        cat_cols: None(default) or specify categorical columns. None means we will figure them out
        numeric_cols: None(default) or specify numeric columns. None means we will figure them out
        Note: Categorical features are by default filled with mode(default), constant value, or Pandas Interpolate
        
        encode_data: Should it encode your categorical data?
        method: 
            available encoders:
                BackwardDifferenceEncoder
                BaseNEncoder
                BinaryEncoder
                CatBoostEncoder
                CountEncoder
                GLMMEncoder
                HashingEncoder
                JamesSteinEncoder
                LeaveOneOutEncoder
                MEstimateEncoder
                OneHotEncoder
                OrdinalEncoder
                SumEncoder
                PolynomialEncoder
                TargetEncoder
                WOEEncoder
                PolynomialWrapper
        thresh: threshold to determine categorical variables
        
        normalize: Should it normalize your data?
        method_transform: method of transformation
          Supported methods:
            'LogTransform'
            'StandardScaler',
            'MinMaxScaler',
            'MaxAbsScaler',
            'QuantileTransformer'
            'Yeo-Johnson'
            'Normalizer'
            'Box-Cox'
        thresh_numeric_transform: threshold for numeric values
        
        remove_outliers: Should it remove outliers?
        qu_fence: inner (1.5*iqr) or outer (3.0*iqr) values: "inner" or "outer"
        
        new_features: Should it make new features along with feature selection?
        task: is it classification task or regression?
        test_data: test dataframe (if any)
        verbose: want to see the progress of data creation?
        feateng_step: the more = more features = more RAM required.
        """
        if fill_missing_data:
            print('='*30)
            print('Handling Missing Data')
            print('='*30)
            handle = False
            if data.isnull().sum().max():
                print('There is missing data')
                res, success, data = self.fill_missing_data(data.copy(), method_cat = method_cat, method_num = method_num, 
                                                            drop = drop, cat_cols = cat_cols, numeric_cols = numeric_cols, 
                                                            thresh_cat = thresh_cat, thresh_numeric = thresh_numeric)
                if success:
                    handle = True
                    print('Missing data Handled')
                else:
                    print(res)
            else:
                print('There is no missing data')
            print('='*30)
            print('\n\n')
                
        if encode_data:
            print('='*30)
            print('Encoding Data')
            print('='*30)
            if cat_cols:
                df, cats = self.Encode(data.copy(), data[label].copy(), cat_cols = cat_cols, method = method, thresh = thresh)
            else:
                df, cats = self.Encode(data.copy(), data[label].copy(), method = method, thresh = thresh)
            data[cats] = df
            print('Data Encoded')
            print('='*30)
            print('\n\n')
            
        if normalize:
            print('='*30)
            print('Transforming Data')
            print('='*30)
            if numeric_cols:
                res, msg, cols = self.transformation(data.copy(), numeric_cols = numeric_cols, method=method_transform, 
                                                     thresh_numeric = thresh_numeric_transform)
            else:
                res, msg, cols = self.transformation(data.copy(), method=method_transform , 
                                                     thresh_numeric = thresh_numeric_transform)
            print('Data Transformed')
            print('='*30)
            print('\n\n')
            
        if remove_outliers:
            print('='*30)
            print('Handling Outliers')
            print('='*30)
            data, fig1, fig2 = self.remove_outliers_using_quantiles(data,label,qu_fence)
            print('='*30)
            print('\n\n')
            
        X = data.copy()
        y = data[label].copy()
        if new_features:
            print('='*30)
            print('Creating New Features with Features Selection')
            print('='*30)
            if task == 'classification':
                afc = AutoFeatClassifier(verbose=1, feateng_steps=feateng_steps)
                try:
                    X , y = data.drop(label,axis=1), data[label]
                except:
                    pass
                X = afc.fit_transform(X, y)
                if not test_data == None:
                    test_data = afc.transform(test_data)
            else:
                afc = AutoFeatRegressor(verbose=1, feateng_steps=feateng_steps)
                X = data.copy()
                try:
                    X , y = data.drop(label,axis=1), data[label]
                except:
                    pass
                X = afc.fit_transform(X, y)
                if not test_data == None:
                    test_data = afc.transform(test_data)
            print('='*30)
            print('\n\n')
        self.X = X
        self.y = y
        self.test_data = test_data
        print('='*30)
        print('Call get_new_data() function to collect new data');
        
    def get_new_data(self):
        return self.X, self.y, self.test_data
    
    def identify_cats(self, data, thresh_cat):
        cats = []
        for col in data.columns:
            if (len(pd.unique(data[col]))/data.shape[0]) < thresh_cat:
                cats.append(col)
        return cats

    def count_plot_categorical(self, data, cols):
        x = len(cols)
        fig = plt.figure(figsize=((x+1)*10,x*8))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(0, len(cols)):
            ax = fig.add_subplot(2, 3, i + 1)
            sns.countplot(data[cols[i]],ax = ax)
            plt.xticks(rotation=45)
        return fig

    def plot_Distribution(self, data, cols):
        x = len(cols)
        fig = plt.figure(figsize=((x+1)*5,x*4))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(0, len(cols)):
            ax = fig.add_subplot(x+1, 3, i + 1)
            sns.distplot(data[cols[i]],ax = ax)
        return fig

    def identify_numeric(self, data, thresh_numeric):
        numeric = []
        for col in data.columns:
            if (len(pd.unique(data[col]))/data.shape[0]) > thresh_numeric:
                numeric.append(col)
        return numeric

    def plot_outliers(self, df, col):
        plt.figure(figsize=(15,10))
        return sns.distplot(df[col])
    
    def fill_missing_data( self,
        data, method_cat = 'Mode', method_num = 'Mean', drop = True, cat_cols = None, 
        numeric_cols = None, thresh_cat = 0.10, thresh_numeric = 0.10):
        """
        data: data 
        method_num: possible methods are: Mean(default), Median, Mode, Constant and Interpolate
        method_cat: possible methods are: Mode(default), Constant and Interpolate
        drop: if the missing ratio is greater than available data, should the feature be dropped(default) or not
        cat_cols: None(default) or specify categorical columns. None means we will figure them out
        numeric_cols: None(default) or specify numeric columns. None means we will figure them out
        Note: Categorical features are by default filled with mode(default), constant value, or Pandas Interpolate
        """
        try:
            if drop:
                for col in data.columns:
                    try:
                        if data[col].isnull().value_counts()[True] > data[col].isnull().value_counts()[False]:
                            data.drop(col,axis=1,inplace=True)
                    except:
                        if data[col].isnull().value_counts()[0] == True:
                            data.drop(col,axis=1,inplace=True)
                        else:
                            continue
            cats = []
            numeric = []
            if not cat_cols == None:
                cats = cat_cols
            if not numeric_cols == None:
                numeric = numeric_cols
            if not cats:
                cats = self.identify_cats(data, thresh_cat)
            if not numeric:
                numeric = self.identify_numeric(data, thresh_numeric)
            if not method_cat == "None":
                for col in cats:
                    if method_cat == 'Mode':
                        data[col].fillna(data[col].mode()[0],inplace=True)
                    elif method_cat == 'Constant':
                        data[col].fillna(-9999,inplace=True)
                    else:
                        data[col].interpolate(method ='linear', limit_direction ='forward', inplace=True)
            if not method_num == "None":
                for col in numeric:
                    data[col] = data[col].astype('float')
                    if method_num == 'Mean':
                        data[col].fillna(data[col].mean(),inplace=True)
                    elif method_num == 'Mode':
                        data[col].fillna(data[col].mode()[0],inplace=True)
                    elif method_num == 'Constant':
                        data[col].fillna(-9999,inplace=True)
                    else:
                        data[col].interpolate(method ='linear', limit_direction ='forward', inplace=True)
            global fill_method, cc, nc, d
            fill_method = (method_cat + ' & ' + method_num if not method_cat == method_num else method_cat)
            cc = cats
            nc = numeric
            d = drop
            return "Success",True,data
        except Exception as e:
            return e, False, data
    
    def Encode(self, data, y = None, method = 'TargetEncoder', cat_cols = None, thresh = 0.10):
        '''
        data: data
        y: target column
        method: 
            available encoders:
                BackwardDifferenceEncoder
                BaseNEncoder
                BinaryEncoder
                CatBoostEncoder
                CountEncoder
                GLMMEncoder
                HashingEncoder
                JamesSteinEncoder
                LeaveOneOutEncoder
                MEstimateEncoder
                OneHotEncoder
                OrdinalEncoder
                SumEncoder
                PolynomialEncoder
                TargetEncoder
                WOEEncoder
                PolynomialWrapper
        '''
        try:
            cats = []
            if cat_cols == None:
                cats = self.identify_cats(data, thresh)
            encoder = None
            if method == 'BackwardDifferenceEncoder':
                encoder = BackwardDifferenceEncoder(cols=[...])
            if method == 'BaseNEncoder':
                encoder = BaseNEncoder(cols=cats)
            if method == 'BinaryEncoder':
                encoder = BinaryEncoder(cols=cats)
            if method == 'CatBoostEncoder':
                encoder = CatBoostEncoder(cols=cats)
            if method == 'CountEncoder':
                encoder = CountEncoder(cols=cats)
            if method == 'GLMMEncoder':
                encoder = GLMMEncoder(cols=cats)
            if method == 'HashingEncoder':
                encoder = HashingEncoder(cols=cats)
            if method == 'HelmertEncoder':
                encoder = HelmertEncoder(cols=cats)
            if method == 'JamesSteinEncoder':
                encoder = JamesSteinEncoder(cols=cats)
            if method == 'LeaveOneOutEncoder':
                encoder = LeaveOneOutEncoder(cols=cats)
            if method == 'MEstimateEncoder':
                encoder = MEstimateEncoder(cols=cats)
            if method == 'OneHotEncoder':
                encoder = OneHotEncoder(cols=cats)
            if method == 'OrdinalEncoder':
                encoder = OrdinalEncoder(cols=cats)
            if method == 'SumEncoder':
                encoder = SumEncoder(cols=cats)
            if method == 'PolynomialEncoder':
                encoder = PolynomialEncoder(cols=cats)
            if method == 'TargetEncoder':
                encoder = TargetEncoder(cols=cats)
            if method == 'WOEEncoder':
                encoder = WOEEncoder(cols=cats)
            if method == 'PolynomialWrapper':
                encoder = PolynomialWrapper(cols=cats)
            global enc_method, e_cc
            enc_method = method
            e_cc = cats
            encoder.fit(data[cats], y)
            print('Success')
            #fig = count_plot_categorical(data, cats)
            return encoder.transform(data[cats]), cats
        except Exception as e:
            return e
    def transformation(self, data, method='LogTransform', numeric_cols = None, thresh_numeric = 0.10):
        '''
        data: data
        method: method of transformation
          Supported methods:
            'LogTransform'
            'StandardScaler',
            'MinMaxScaler',
            'MaxAbsScaler',
            'QuantileTransformer'
            'Yeo-Johnson'
            'Normalizer'
            'Box-Cox'
        numeric_cols: numerical columns as list
        thresh_numeric: threshold for numeric values
        '''
        try:
            num_c = []
            if numeric_cols == None:
                num_c = self.identify_numeric(data, thresh_numeric)
            table1 = plot_table(data, num_c)
            scaler = None
            if method == 'LogTransform':
                data[num_c] = np.log1p(data[num_c])
            if method == 'StandardScaler':
                scaler = StandardScaler()
            if method == 'MinMaxScaler':
                scaler = MinMaxScaler()
            if method == 'MaxAbsScaler':
                scaler = MaxAbsScaler()
            if method == 'QuantileTransformer':
                scaler = QuantileTransformer()
            if method == 'Yeo-Johnson':
                scaler = PowerTransformer()
            if method == 'Normalizer':
                scaler = Normalizer()
            if method == 'Box-Cox':
                scaler = PowerTransformer(method='box-cox')
            if not method == 'LogTransform':
                data[num_c] = scaler.fit_transform(data[num_c])
            return "Success", data, num_c
        except Exception as e:
            return e, data, num_c

    def remove_outliers_using_quantiles(self, qu_dataset, qu_field, qu_fence = 'inner'):
        '''
        Function: remove_outliers_using_quantiles(qu_dataset, qu_field, qu_fence)
          1- Remove outliers according to the given fence value and return new dataframe.
          2- Print out the following information about the data
             - interquartile range
             - upper_inner_fence
             - lower_inner_fence
             - upper_outer_fence
             - lower_outer_fence
             - percentage of records out of inner fences
             - percentage of records out of outer fences
        Input: 
          - pandas dataframe (qu_dataset)
          - name of the column to analyze (qu_field)
          - inner (1.5*iqr) or outer (3.0*iqr) (qu_fence) values: "inner" or "outer"
        Output:
          - new pandas dataframe (output_dataset)
        '''
        global iqr
        global upper_inner_fence
        global lower_inner_fence 
        global upper_outer_fence 
        global lower_outer_fence 
        global percentage1 
        global percentage2 
        global len_ds 
        global len_out
        print('Before outlier removal ')
        #fig1 = plot_outliers(qu_dataset, qu_field)
        #plt.show()
        a = qu_dataset[qu_field].describe()

        iqr = a["75%"] - a["25%"]
        print("interquartile range:", iqr)

        upper_inner_fence = a["75%"] + 1.5 * iqr
        lower_inner_fence = a["25%"] - 1.5 * iqr
        print("upper_inner_fence:", upper_inner_fence)
        print("lower_inner_fence:", lower_inner_fence)

        upper_outer_fence = a["75%"] + 3 * iqr
        lower_outer_fence = a["25%"] - 3 * iqr
        print("upper_outer_fence:", upper_outer_fence)
        print("lower_outer_fence:", lower_outer_fence)

        count_over_upper = len(qu_dataset[qu_dataset[qu_field]>upper_inner_fence])
        count_under_lower = len(qu_dataset[qu_dataset[qu_field]<lower_inner_fence])
        percentage1 = 100 * (count_under_lower + count_over_upper) / a["count"]
        print("percentage of records out of inner fences: %.2f"% (percentage1))

        count_over_upper = len(qu_dataset[qu_dataset[qu_field]>upper_outer_fence])
        count_under_lower = len(qu_dataset[qu_dataset[qu_field]<lower_outer_fence])
        percentage2 = 100 * (count_under_lower + count_over_upper) / a["count"]
        print("percentage of records out of outer fences: %.2f"% (percentage2))

        if qu_fence == "inner":
            output_dataset = qu_dataset[qu_dataset[qu_field]<=upper_inner_fence]
            output_dataset = output_dataset[output_dataset[qu_field]>=lower_inner_fence]
        elif qu_fence == "outer":
            output_dataset = qu_dataset[qu_dataset[qu_field]<=upper_outer_fence]
            output_dataset = output_dataset[output_dataset[qu_field]>=lower_outer_fence]
        else:
            output_dataset = qu_dataset
        len_ds = len(qu_dataset)
        print("length of input dataframe:", len_ds)
        len_out = len(output_dataset)
        print("length of new dataframe after outlier removal:", len_out)
        print('After outlier removal ')
        #fig2 = plot_outliers(qu_dataset, qu_field)
        #plt.show()
        return output_dataset, None, None

    