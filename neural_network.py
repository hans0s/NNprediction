#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import configparser as cf
import string, os, sys
from sklearn.preprocessing import scale
from keras.layers import Input, Dense, LSTM, merge
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.convolutional import Conv2D, Conv1D
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adamax
from keras.callbacks import ReduceLROnPlateau
import json

#——————————————————读取自动开关——————————————————
f_sw = open("auto_cfg\\auto_switch.txt", "r")
f2 = f_sw.read()
auto_switch = int(f2)
print("Auto switched to:",auto_switch)

#——————————————————读取配置文件——————————————————
if auto_switch == 0:
    print("Please input config filename:")
    config_file=input()
if auto_switch == 1:
    f_list = open("auto_cfg\\auto_list.txt", "r")
    f3 = f_list.readlines()
    for i in range(len(f3)-1):
        f3[i]=f3[i].strip()
    print("Auto list:",f3)
    auto_list_size = len(f3)
    f_index = open("auto_cfg\\auto_index.txt", "r")
    f4 = f_index.read()
    f_index.close()
    auto_index = int(f4)
    config_file=str(f3[auto_index])
config=cf.ConfigParser()
config.read("data\\"+config_file) #读取输入的配置文件
print("Config file:",config_file,"has been loaded.")

#——————————————————参数赋值——————————————————
train_file=config.get('file','train_file') #训练文件
train_begin_cfg=config.getint('data','train_begin') #训练开始点
train_end_cfg=config.getint('data','train_end') #训练结束点
test_file=config.get('file','test_file') #测试文件
test_begin_cfg=config.getint('data','test_begin') #测试开始点
test_end=config.getint('data','test_end') #测试结束点
input_size=config.getint('data','input_size') #输入长度
output_size=config.getint('other','output_size') #输出长度
time_step_cfg=config.getint('data','time_step') #时间长度
label=config.getint('data','label') #Y值编号
train_method=0
pred_method=config.getint('other','pred_method') #测试方法编号
model_output=config.get('file','model_output') #输出模型的编号
model_input=config.get('file','model_input') #输入模型的编号
random=config.getint('other','random')
pred_len=5
nor_shift_len=config.getint('nor','nor_shift_len')
nor_method=config.getint('nor','nor_method')
nor_len_set=config.getint('nor','nor_len_set')
nor_method_plus=config.getint('nor','nor_method_plus')
index_size=config.getint('data','index_size')
nextdays=config.getint('data','nextdays')
mm_customize=config.getint('nor','mm_customize')
y_nor=config.getint('nor','y_nor')

with open("network.json") as file:  # 读取network文件
    config_nw = json.load(file)
op=config_nw["compile"]["optimizer"]
batch_size_cfg=int(config_nw["train"]["batch_size"]) #训练分批大小

#——————————————————最大最小标准化自定义参数选项——————————————————
mm_data = []
mm_data_x = []
mm_data_y = []
if mm_customize == 1:
    f_mm=open("data\\"+train_file+".mm.csv")
    df=pd.read_csv(f_mm)     #读入数据文件
    data=df.iloc[:,2:(input_size+3+label)].values  #取第3-n列
    mm_data = data
    mm_data_x = data[:,:input_size]
    mm_data_y = data[:,input_size+label]

#——————————————————通过keras定义神经网络——————————————————
#——————————————————构建神经网络——————————————————
def build_nn(train_x,train_y,point,config):
    model=Sequential()

    for layer_number, layer in enumerate(config["layers"]):
        if layer["type"] == "First_Dropout":
            model.add(Dropout(layer["dropout_in"], input_shape=(time_step_cfg, input_size)))
        elif layer["type"] == "LSTM":
            model.add(LSTM(int(layer["neuron_number"]), return_sequences=bool(layer["return_sequences"]), activation=layer["activation_function"]))  # LSTM层
            model.add(Dropout(layer["dropout_out"]))  # dropout
        elif layer["type"] == "RNN":
            model.add(SimpleRNN(int(layer["neuron_number"]), return_sequences=bool(layer["return_sequences"]), activation=layer["activation_function"]))  # RNN层
        elif layer["type"] == "Conv1D":
            model.add(Conv1D(filters=layer["filter_number"], kernel_size=layer["filter_shape"], strides=layer["strides"], padding=layer["padding"], activation=layer["activation_function"]))  # CNN层
        elif layer["type"] == "Conv2D":
            model.add(Conv2D(filters=layer["filter_number"], kernel_size=layer["filter_shape"], strides=layer["strides"], padding=layer["padding"], activation=layer["activation_function"]))  # CNN层
        elif layer["type"] == "Dense":
            model.add(Dense(int(layer["neuron_number"]), activation=layer["activation_function"])) # 连接
        elif layer["type"] == "Output_Dense":
            model.add(Dense(output_size, activation=layer["activation_function"])) # 输出

    model.compile(loss=config["compile"]["loss_function"], optimizer=config["compile"]["optimizer"],
                  metrics=[config["compile"]["metrics"]])

    callback_list = [
        ReduceLROnPlateau(monitor='val_loss',  # 监控模型的验证损失
                          factor=0.5,  # 触发时将学习率除以10
                          patience=50)  # 如果验证损失在10轮内都没有改善，那么就触发这个回调函数
    ]

    history = model.fit(train_x, train_y, epochs=int(config["train"]["epochs"]),
                        batch_size=config["train"]["batch_size"], verbose=int(config["train"]["verbose"]),
                        validation_split=float(config["train"]["validation_split"]), callbacks=callback_list)

    if point < 3:
        model.save_weights("model\\trend\\"+str(train_method)+"_"+str(op)+"_"+model_output+"_T"+str(time_step_cfg)+"_label_"+str(label)+"_N"+str(nor_method)+"_"+"P"+str(point)+"_weights.h5") #保存weights
        model.save("model\\trend\\"+str(train_method)+"_"+str(op)+"_"+model_output+"_T"+str(time_step_cfg)+"_label_"+str(label)+"_N"+str(nor_method)+"_"+"P"+str(point)+"_model.h5") #保存模型
    if point >= 3:
        model.save_weights("model\\"+str(train_method)+"_"+str(op)+"_"+model_output+"_T"+str(time_step_cfg)+"_label_"+str(label)+"_N"+str(nor_method)+"_weights.h5") #保存weights
        model.save("model\\"+str(train_method)+"_"+str(op)+"_"+model_output+"_T"+str(time_step_cfg)+"_label_"+str(label)+"_N"+str(nor_method)+"_model.h5") #保存模型
    #打印loss和acc
    print("loss:",history.history['loss'])
    if float(config["train"]["validation_split"]) >0:
        print("val_loss:",history.history['val_loss'])
    #图形表示
    loss = history.history['loss']
    if float(config["train"]["validation_split"]) >0:
        val_loss = history.history['val_loss']
    plt.figure()
    plt.plot(loss[5:])
    if float(config["train"]["validation_split"]) >0:
        plt.plot(val_loss[5:])
    plt.title('train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper right')
    if point < 3:
        plt.savefig("model\\trend\\"+str(train_method)+"_"+str(op)+"_"+model_output+"_T"+str(time_step_cfg)+"_label_"+str(label)+"_N"+str(nor_method)+"_"+"P"+str(point)+"_model.png", dpi=200)
    if point >= 3:
        plt.savefig("model\\"+str(train_method)+"_"+str(op)+"_"+model_output+"_T"+str(time_step_cfg)+"_label_"+str(label)+"_N"+str(nor_method)+"_model.png", dpi=200)
    if point >= 3 and auto_switch == 0:
        plt.show()
    #保存所用的配置文件信息
    f = open("data\\" + config_file, "r")
    f1 = f.readlines()
    s0 = pd.Series(f1)
    save = pd.DataFrame(s0)
    if point < 3:
        save.to_html("model\\trend\\"+str(train_method)+"_"+str(op)+"_"+model_output+"_T"+str(time_step_cfg)+"_label_"+str(label)+"_N"+str(nor_method)+"_"+"P"+str(point)+"_model.html")
    if point >= 3:
        save.to_html("model\\"+str(train_method)+"_"+str(op)+"_"+model_output+"_T"+str(time_step_cfg)+"_label_"+str(label)+"_N"+str(nor_method)+"_model.html")
    return model