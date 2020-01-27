import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_squared_error

from scipy import stats

class NumericalVis():
    def __init__(self, df, h_limit=4, char_width=24, style="ticks"):
        num_cols = list(df.select_dtypes(include=[np.number]).columns.values)
        self.num_cols_mtx = [num_cols[i:i+h_limit] for i in range(0, len(num_cols), h_limit)]
        self.v_limit = len(self.num_cols_mtx)
        self.h_limit = h_limit
        self.style = style
        frm = lambda x: '{:,.4f}'.format(x).rstrip('0').rstrip('.').rjust(char_width)
        frm_cat = lambda x: '{}'.format(x).rjust(char_width)
        arrays = {}
        for num_col in [item for sublist in self.num_cols_mtx for item in sublist]:
            ser = df[num_col]
            arr = ser.fillna(np.NaN).to_numpy()
            desc = {
                'Name': frm_cat(num_col),
                'Logical Type': frm_cat('num'),
                'Storage Type': frm_cat(ser.dtype.name),
                'Count': frm(arr.size),
                'Missing': frm(np.where((arr != arr) | (arr == None))[0].size),
                'Mean': frm(np.nanmean(arr)),
                'Min': frm(np.nanmin(arr)),
                'Max': frm(np.nanmax(arr)),
                'Median': frm(np.nanmedian(arr)),
                'Stdev': frm(np.nanstd(arr)),
                'Unique': frm(np.unique(arr).size)
            }
            arrays[num_col] = {
                'ser': ser,
                'arr': arr,
                'desc': desc      
            }
        self.arrays = arrays         
    def plot(self):
        sns.set()
        sns.set_style(self.style)
        v_limit, h_limit = self.v_limit, self.h_limit
        self.fig, self.axs = plt.subplots(2*v_limit, 1+h_limit, figsize=(2+4*h_limit, 8*v_limit),
                                gridspec_kw={'height_ratios': [1,3]*v_limit,
                                             'width_ratios': [2]+[4]*h_limit})
        for idx, row in enumerate(self.num_cols_mtx):
            ax = self.axs[2*idx+0][0]
            ax.axis('off')
            ax.text(0, 1, 'Numerical', transform=ax.transAxes, fontsize=15, family='monospace', fontweight='bold')
            #ax.axis('off')
            ax = self.axs[2*idx+1][0]
            dic = self.arrays[self.num_cols_mtx[0][0]]['desc']
            for key_idx, key in enumerate(dic):
                ax.text(0.01, 1-0.09*key_idx, key, transform=ax.transAxes, fontsize=13, family='monospace')
            ax.axis('off')
            for num_col_idx, num_col in enumerate(row):
                ax = self.axs[2*idx+0][num_col_idx+1]
                arr = self.arrays[num_col]['arr']
                arr_non_nan = arr[~np.isnan(arr)]
                ax.hist(arr_non_nan)
                ax.spines["top"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.tick_params(labelsize=10) 
                dic = self.arrays[num_col]['desc']
                ax = self.axs[2*idx+1][num_col_idx+1]
                for key_idx, key in enumerate(dic):
                    ax.text(0.01, 1-0.09*key_idx, dic[key], transform=ax.transAxes, fontsize=13, family='monospace')
                ax.axis('off')   
            for xx in range(num_col_idx+2, h_limit+1):
                self.axs[2*idx+0][xx].axis('off')
                self.axs[2*idx+1][xx].axis('off')

class CategoricalVis():
    def __init__(self, df, h_limit=4, char_width=24, style="ticks"):
        num_cols = list(df.select_dtypes(exclude=[np.number]).columns.values)
        self.num_cols_mtx = [num_cols[i:i+h_limit] for i in range(0, len(num_cols), h_limit)]
        self.v_limit = len(self.num_cols_mtx)
        self.h_limit = h_limit
        self.style = style
        frm = lambda x: '{:,.4f}'.format(x).rstrip('0').rstrip('.').rjust(char_width)
        frm_cat = lambda x: '{}'.format(x).rjust(char_width)
        arrays = {}
        for num_col in [item for sublist in self.num_cols_mtx for item in sublist]:
            ser = df[num_col]
            arr = ser.fillna(np.NaN).astype(str).to_numpy(dtype=object)
#             (values, counts) = np.unique(arr, return_counts=True)
#             ind = np.argmax(counts)
#             most_frequent = values[ind]
            counts = ser.value_counts()
            ind_max = counts.idxmax()
            most_frequent = ind_max
            most_frequent_count = counts[ind_max]
            ind_min = counts.idxmin()
            less_frequent = ind_min
            less_frequent_count = counts[ind_min]            
            most_frequent_per = '{:,.2f}'.format(most_frequent_count/np.sum(counts)*100).rstrip('0').rstrip('.')               
            desc = {
                'Name': frm_cat(num_col),
                'Logical Type': frm_cat('num'),
                'Count': frm(len(ser)),
                'Unique': frm(len(ser.unique())), 
                'Most Freq.': frm_cat(most_frequent + ' (' + most_frequent_per+'%)'),
                'Max Freq.': frm(most_frequent_count),
                'Min Freq.': frm(less_frequent_count)
            }
            arrays[num_col] = {
                'ser': ser,
                'arr': arr,
                'desc': desc      
            }
        self.arrays = arrays         
    def plot(self):
        sns.set()
        sns.set_style(self.style)
        v_limit, h_limit = self.v_limit, self.h_limit
        self.fig, self.axs = plt.subplots(2*v_limit, 1+h_limit, figsize=(2+4*h_limit, 8*v_limit),
                                gridspec_kw={'height_ratios': [1,3]*v_limit,
                                             'width_ratios': [2]+[4]*h_limit})
        for idx, row in enumerate(self.num_cols_mtx):
            ax = self.axs[2*idx+0][0]
            ax.axis('off')
            ax.text(0, 1, 'Categorical', transform=ax.transAxes, fontsize=15, family='monospace', fontweight='bold')
            ax = self.axs[2*idx+1][0]
            dic = self.arrays[self.num_cols_mtx[0][0]]['desc']
            for key_idx, key in enumerate(dic):
                ax.text(0.01, 1-0.09*key_idx, key, transform=ax.transAxes, fontsize=13, family='monospace')
            ax.axis('off')
            for num_col_idx, num_col in enumerate(row):
                ax = self.axs[2*idx+0][num_col_idx+1]
                dic = self.arrays[num_col]['desc']
                if len(self.arrays[num_col]['ser'].unique()) < 10000:
                    ax.hist(self.arrays[num_col]['arr'])
                else:
                    ax.hist([1])
                if len(self.arrays[num_col]['ser'].unique()) > 10:
                    ax.xaxis.set_major_locator(plt.NullLocator())
                ax.spines["top"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.tick_params(labelsize=10) 
                ax = self.axs[2*idx+1][num_col_idx+1]
                for key_idx, key in enumerate(dic):
                    ax.text(0.01, 1-0.09*key_idx, dic[key], transform=ax.transAxes, fontsize=13, family='monospace')
                ax.axis('off')   
            for xx in range(num_col_idx+2, h_limit+1):
                self.axs[2*idx+0][xx].axis('off')
                self.axs[2*idx+1][xx].axis('off')

class Details():
    def __init__(self, df, target_column, random_state=32):
        num_cols = list(df.select_dtypes(include=[np.number]).columns.values)
        cat_cols = list(df.select_dtypes(exclude=[np.number]).columns.values)
        self.cols = [i for i in df.columns if i != target_column]
        self.num_cols = [i for i in num_cols if i != target_column]
        self.cat_cols = [i for i in cat_cols if i != target_column]
        self.target_column = target_column
        self.target_values = df[target_column].unique()
        self.df = df
        self.random_state = random_state

    def importance(self, current_column):
        target_column = self.target_column
        df = self.df
        data = df[[current_column, target_column]].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=[target_column]),
            data[target_column],
            test_size=0.2,
            stratify=data[target_column],
            random_state=self.random_state
        )

        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        transformers = [
            ('num', num_pipe, [current_column])
        ]

        preprocessor = ColumnTransformer(transformers = transformers)

        ml_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver = 'lbfgs', C=0.5, max_iter=1000))
        ])

        ml_pipe.fit(X_train, y_train)
        score = roc_auc_score(y_test, ml_pipe.predict_proba(X_test)[:, 1])
        return score

    def importance2(self, current_column):
        target_column = self.target_column
        df = self.df
        data = df[[current_column, target_column]].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=[target_column]),
            data[target_column],
            test_size=0.3,
            random_state=self.random_state
        )

        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        transformers = [
            ('num', num_pipe, [current_column])
        ]

        preprocessor = ColumnTransformer(transformers = transformers)

        ml_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LinearRegression())
        ])

        ml_pipe.fit(X_train, y_train)
        #score = roc_auc_score(y_test, ml_pipe.predict(X_test))
        score = np.sqrt(mean_squared_error(y_test, ml_pipe.predict(X_test)))
        return score

    def corr(self):
        df = self.df
        sns.set()
        plt.figure(figsize=(16, 10))
        sns.heatmap(df[self.num_cols].corr(), annot=True, fmt=".2f")
        b, t = plt.ylim()
        b += 0.5
        t -= 0.5
        plt.ylim(b, t)
        fig = plt.figure()
        fig.tight_layout()
        plt.show()

    def pairplot(self):
        df = self.df
        sns.set()
        plt.figure(figsize=(16, 10))
        sns.pairplot(df[self.num_cols])
        fig = plt.figure()
        fig.tight_layout()
        plt.show()

    def show_details(self, columns = []):
        df = self.df
        sns.set()
        target_column = self.target_column

        def frm(x):
            if isinstance(x, (int, float, complex)) and not isinstance(x, bool):
                result = '{:,.4f}'.format(x).rstrip('0').rstrip('.')
            else:
                result = '{}'.format(x)
            return result

        for current_column in self.cols:
            if len(columns) > 0 and current_column not in columns:
                continue
            print(('-' * 60 + '\n') * 3)
            if current_column in self.num_cols:
                if len(self.target_values) <= 2:
                    importance = self.importance(current_column)
                    ser = df.loc[:, current_column]
                    ser_non_na = ser.fillna(ser.median())
                    ser_0 = df.loc[df[target_column] == 0, current_column]
                    ser_0_non_na = ser_0.fillna(ser_0.median())
                    ser_1 = df.loc[df[target_column] == 1, current_column]
                    ser_1_non_na = ser_1.fillna(ser_1.median())
                    arr = ser.fillna(np.NaN).to_numpy()
                    arr_non_na = ser_non_na.to_numpy()

                    fig, ax = plt.subplots(1, 4, figsize=(15, 3))
                    fig.tight_layout()
                    sns.distplot(ser_non_na, ax=ax[0])
                    sns.distplot(ser_0_non_na, label='asd', ax=ax[1])
                    sns.distplot(ser_1_non_na, label='ttt', ax=ax[1])
                    sns.boxplot(x=target_column, y=current_column, data=df, ax=ax[2])
                    sns.boxplot(y=current_column, data=df, ax=ax[3])
                    ax[1].legend(labels=[0, 1])
                    fig.show()

                else:
                    importance = self.importance2(current_column)
                    ser = df.loc[:, current_column]
                    ser_non_na = ser.fillna(ser.median())
                    ser_0 = df.loc[df[target_column] == 0, current_column]
                    ser_0_non_na = ser_0.fillna(ser_0.median())
                    ser_1 = df.loc[df[target_column] == 1, current_column]
                    ser_1_non_na = ser_1.fillna(ser_1.median())
                    arr = ser.fillna(np.NaN).to_numpy()
                    arr_non_na = ser_non_na.to_numpy()

                    fig, ax = plt.subplots(1, 4, figsize=(15, 3))
                    fig.tight_layout()
                    sns.distplot(ser_non_na, ax=ax[0])
                    sns.scatterplot(x=target_column, y=current_column, data=df, ax=ax[1])
                    sns.boxplot(y=current_column, data=df, ax=ax[3])
                    fig.show()

                plt.pause(0.5)

                desc = {
                    'Name': current_column,
                    'Logical Type': 'num',
                    'Storage Type': ser.dtype.name,
                    'Count': arr.size,
                    'Missing': np.where((arr != arr) | (arr == None))[0].size,
                    'Mean': np.nanmean(arr),
                    'Min': np.nanmin(arr),
                    'Max': np.nanmax(arr),
                    'Median': np.nanmedian(arr),
                    'Stdev': np.nanstd(arr),
                    'Unique': np.unique(arr).size,
                    'Importance': importance
                }

                arr_clipped = np.clip(arr, 0.01, None)
                mean_clipped = np.nanmean(arr_clipped)
                skew = {
                    'Skew:': '',
                    'Init': stats.skew(arr),
                    'Log': stats.skew(np.log(arr_clipped)),
                    'Adj. Log 0.001': stats.skew(np.log(arr_clipped/mean_clipped + 0.001)),
                    'Adj. Log 0.01': stats.skew(np.log(arr_clipped/mean_clipped + 0.01)),
                    'Adj. Log 0.1': stats.skew(np.log(arr_clipped/mean_clipped + 0.1)),
                    'Adj. Log 0.5': stats.skew(np.log(arr_clipped/mean_clipped + 0.5)),
                    'Cbrt': stats.skew(np.cbrt(np.absolute(arr))),
                    'Sqrt': stats.skew(np.sqrt(np.absolute(arr))),
                    'Pos. Reciprocal': stats.skew(1/arr_clipped),
                    'Neg. Reciprocal': stats.skew(-1/arr_clipped),
                }

                column_size = 25

                desc_keys = list(desc.keys())
                skew_keys = list(skew.keys())
                keys_num = max(len(desc_keys), len(skew_keys))
                for i in range(0, keys_num):
                    if i < len(desc_keys):
                        col1 = desc_keys[i] + frm(desc.get(desc_keys[i])).rjust(column_size - len(desc_keys[i]))
                    else:
                        col1 = ''.rjust(column_size)

                    if i < len(skew_keys):
                        col2 = skew_keys[i] + frm(skew.get(skew_keys[i])).rjust(column_size - len(skew_keys[i]))
                    else:
                        col2 = ''.rjust(column_size)
                    line = col1 + ''.rjust(8) + col2
                    print(line)

            if current_column in self.cat_cols:
                ser = df[current_column]
                counts = ser.value_counts()
                column_size = 30

                desc = {
                    'Name': frm(current_column),
                    'Logical Type': frm('cat'),
                    'Count': frm(len(ser)),
                    'Missing': frm(ser.isna().sum()),
                    'Unique': frm(len(ser.unique())),
                }
                desc_keys = list(desc.keys())
                keys_num = len(desc_keys)
                for i in range(0, keys_num):
                    if i < len(desc_keys):
                        col1 = desc_keys[i] + frm(desc.get(desc_keys[i])).rjust(column_size - len(desc_keys[i]))
                    else:
                        col1 = ''.rjust(column_size)
                    line = col1
                    print(line)

                rows_num = 8
                counts_top = counts[:rows_num]
                counts_bottom = counts[rows_num:][-rows_num:]
                counts_top_len = len(counts_top)
                counts_bottom_len = len(counts_bottom)
                counts_others = len(counts) - counts_top_len - counts_bottom_len
                keys_num = max(counts_top_len, counts_bottom_len)
                for i in range(0, keys_num):
                    if i < counts_top_len:
                        col1 = counts_top.index[i] + frm(counts_top.iloc[i]).rjust(column_size - len(counts_top.index[i]))
                    else:
                        col1 = ''.rjust(column_size)

                    if i < counts_bottom_len:
                        col2 = counts_bottom.index[i] + frm(counts_bottom.iloc[i]).rjust(column_size - len(counts_bottom.index[i]))
                    else:
                        col2 = ''.rjust(column_size)
                    if counts_others > 0:
                        if i == 0:
                            colm = (frm(counts_others) + ' value(s)').rjust(column_size)
                        else:
                            colm = '...'.rjust(column_size)
                    else:
                        colm = ''
                    line = col1 + colm + ''.rjust(8) + col2
                    print(line)