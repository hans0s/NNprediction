import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#参数
index_size = 1
data_size = 6

#输入文件名
print("File 1 name:")
file_1 = input()
#file_1 = "601069.csv"
print("File 2 name:")
file_2 = input()
#file_2 = "gold.csv"

#打开原始文件
f1 = open("raw_data\\"+file_1)
df1 = pd.read_csv(f1)
f2 = open("raw_data\\"+file_2)
df2 = pd.read_csv(f2)

#取文件1的日期集合
data1_date = df1.iloc[:,index_size].values

#匹配文件1日期集合的文件2的各行 -》 新文件2
new_df2 = df2[df2.date.isin(data1_date)]
new_df2 = new_df2.reset_index(drop=True)
print(new_df2)
#new_df2.to_csv("raw_data\\"+"new"+file_2, index=False)

#取新文件2的日期集合
data2_date = new_df2.iloc[:,index_size].values

#匹配新文件2日期集合的文件1的各行 -》 新文件1
new_df1 = df1[df1.date.isin(data2_date)]
new_df1 = new_df1.reset_index(drop=True)
print(new_df1)
#new_df1.to_csv("raw_data\\"+"new"+file_1, index=False)

#合并结果
result = pd.concat([new_df1, new_df2], axis=1)
print(result)
result.to_csv("raw_data\\"+file_1+"+"+file_2, index=False)
