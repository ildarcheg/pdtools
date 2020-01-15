import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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