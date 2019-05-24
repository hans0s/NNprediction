#coding=utf-8

import numpy as np
import pandas as pd
import neural_network
import json
import sys



#——————————————————读入参数——————————————————
input_size=neural_network.input_size
train_file=neural_network.train_file
train_begin=neural_network.train_begin_cfg
train_end=neural_network.train_end_cfg
time_step=neural_network.time_step_cfg
model_output=neural_network.model_output
label=neural_network.label
train_method=neural_network.train_method
random=neural_network.random
nor_method=neural_network.nor_method
op=neural_network.op
mm_customize=neural_network.mm_customize
index_size=neural_network.index_size


#——————————————————方法：获取训练集——————————————————————
def get_train_data():
    # 读取训练文件
    f = open("data\\" + train_file)
    df = pd.read_csv(f)  # 读入数据文件
    data = df.iloc[:, index_size:(input_size + index_size + label + 1)].values  # 取第3-n列
    data_train=data[train_begin-2:train_end-1] #按配置文件取值，从n1行至n2行
    # 结束
    # 打印信息1
    print("Debug information:") #打印debug信息
    print("The len of original train data:", len(data_train))
    print("The first item in original train:",data_train[0])
    print("The last item in original train:",data_train[len(data_train)-1])
    # 结束
    # 数据标准化
    if nor_method == 0: #使用标准差标准化方法
        normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # xy标准化
    if nor_method == 1: #使用最大最小值标准化方法
        if mm_customize ==0: #不使用自定义数据来标准化
            normalized_train_data = (data_train - np.min(data_train, axis=0)) / (np.max(data_train, axis=0)-np.min(data_train, axis=0))  # xy标准化
        if mm_customize ==1: #使用自定义数据来标准化
            normalized_train_data = (data_train - neural_network.mm_data[0]) / (neural_network.mm_data[1]-neural_network.mm_data[0])  # xy标准化
    if nor_method == -1: #不做标准化
        normalized_train_data = data_train
    # 结束
    # 构造训练集
    train_x,train_y=[],[]   #训练集
    train_xy=[]
    if train_method == 0: #训练方法0，默认
        for i in range(len(normalized_train_data) - time_step + 1): #构造训练数据，滚动取值
            xy = normalized_train_data[i:i+time_step]
            train_xy.append(xy.tolist())
        train_xy=np.array(train_xy)
        if random == 1: #打乱数据顺序
            np.random.shuffle(train_xy)
        train_x=train_xy[:,:,:input_size] #训练集X，滚动取值
        train_y0 = train_xy[:, :, input_size + label]
        print("Len of train_y0:",len(train_y0))
        print("train_y0's sharp:",np.shape(train_y0))
        train_y = train_y0[:,time_step-1] #训练集Y，连续取值
        print("Len of train_y:",len(train_y))
        print("train_y's sharp:",np.shape(train_y))
    # 打印信息2
    print("The len of final train data x:",len(train_x))
    print("The len of final train data y:",len(train_y))
    print("Train X's sharp:",np.shape(train_x))
    print("Train Y's sharp:",np.shape(train_y))
    print("Train Y:", train_y)
    # 结束
    return train_x,train_y

#————————————————训练————————————————————
def train_nn():
    # 打印信息0
    print(neural_network.config_file,"Training is being started now...")
    if nor_method == -1:
        print("Train data is without normalization.")
    if nor_method == 0:
        print("Train data is normalized by: Z-Score.")
    if nor_method == 1:
        print("Train data is normalized by: Min-Max...")
        if mm_customize == 0:
            print("...without customized MM.")
        if mm_customize == 1:
            print("...with customized MM:")
            print("(1). Min:",neural_network.mm_data[0])
            print("(2). Max:",neural_network.mm_data[1])
    # 结束
    # 训练
    train_x,train_y=get_train_data() #获取训练数据
    with open("network.json") as file:
        config = json.load(file)
    neural_network.build_nn(train_x,train_y,9,config["layers"]) #开始训练
    # 结束
    # 打印信息3
    print("### The train has been finished: ###") #打印结果
    print("(1).Data file:",train_file,"(from:",train_begin,"to:",train_end,")")
    print("(2).Prediction label:",label)
    print("(3).Model saved to:"+"model\\"+str(train_method)+"_"+str(op)+"_"+model_output+"_T"+str(time_step)+"_label_"+str(label)+"_N"+str(nor_method)+"_xxx.h5")
    print("---------------------------End---------------------------------")
    print("")
    # 结束


#————————————————执行————————————————————
train_nn()