import pandas as pd
import sys
import math

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def downcast(data):
    data_int = data.select_dtypes(include=['int'])
    converted_int = data_int.apply(pd.to_numeric,downcast='unsigned')
    data_float = data.select_dtypes(include=['float'])
    converted_float = data_float.apply(pd.to_numeric,downcast='float')
    #data_obj = data.select_dtypes(include=['object']).copy()
    optimized_data = data.copy()
    optimized_data[converted_int.columns] = converted_int
    optimized_data[converted_float.columns] = converted_float
    return optimized_data

def categorize(data):
    data_obj = data.select_dtypes(include=['object'])
    optimized_data = data.copy()
    for col in data_obj.columns:
        num_unique_values = len(data_obj[col].unique())
        num_total_values = len(data_obj[col])
        if num_unique_values / num_total_values < 0.5:
            optimized_data.loc[:,col] = data_obj[col].astype('category')
        else:
            optimized_data.loc[:,col] = data_obj[col]
    return optimized_data

def read_csv(*args, **kwargs):
    
    def print_pipes(bar_lenth, num, chunksize, row_num):
        current_process = min(bar_lenth,math.floor((num-1)*chunksize/row_num*bar_lenth))
        new_process = min(bar_lenth,math.floor(num*chunksize/row_num*bar_lenth))
        n = new_process - current_process
        for i in range(n):
            print('|' , sep='',end ='', file = sys.stdout , flush = True)
    
    chunksize = kwargs.get('chunksize')
    filepath_or_buffer = kwargs.get('filepath_or_buffer')
    compress = kwargs.get('compress')
    kwargs.pop('compress', None)
    if chunksize:
        if not filepath_or_buffer:
            filepath_or_buffer = args[0]
        row_num = sum(1 for i in open(filepath_or_buffer, 'rb'))
        bar_lenth = 50
        print('Reading chunks of csv...')
        print('-'*(bar_lenth))
        TextFileReader = pd.read_csv(*args, **kwargs)
        dfList = []
        for df in TextFileReader:
            dfList.append(df)
            print_pipes(bar_lenth, len(dfList), chunksize, row_num)
        print('')
        df = pd.concat(dfList, sort=False)
    else:
        print('Reading csv...')
        df = pd.read_csv(*args, **kwargs)
    print('Calculating memory usage...')
    print('Memory usage', mem_usage(df))
    if compress:
        df = downcast(df)
        print('Memory usage after downcasting', mem_usage(df))
        df = categorize(df)
        print('Memory usage after categorization', mem_usage(df))
    
    print('Done.')
    return df