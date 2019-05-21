#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neural_network
from keras.models import load_model
import time

#——————————————————读入参数——————————————————
input_size=neural_network.input_size
test_file=neural_network.test_file
test_begin=neural_network.test_begin_cfg
time_step=neural_network.time_step_cfg
model_input=neural_network.model_input
label=neural_network.label
batch_size_cfg=neural_network.batch_size_cfg
test_end=neural_network.test_end
train_method=neural_network.train_method
pred_method=neural_network.pred_method
pred_len=neural_network.pred_len
train_begin=neural_network.train_begin_cfg
train_end=neural_network.train_end_cfg
nor_method=neural_network.nor_method
nor_len_set=neural_network.nor_len_set
nor_shift_len=neural_network.nor_shift_len
op=neural_network.op
nor_method_plus=neural_network.nor_method_plus
mm_customize=neural_network.mm_customize
index_size=neural_network.index_size
birthday=0 #选1：输出的预测P 不做反标准化
nextdays=neural_network.nextdays #Y是几天以后的值

#——————————————————菜单 - 选择 预测方法 和 交易策略——————————————————
if neural_network.auto_switch == 0:
    print()
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@ Welcome to Prediction Model !!! @@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("(0).X Shift + Y Series")
    print("(1).X Block + Y Series (++)")
    print("(2).to be added...")
    print("(3).X Shift + Y Shift (+++)")
    print("(4).Fixed - X Shift + Y Series (+++++)")
    pred_method = 4
    print("Note:")
    print("Prediction method has been selected to (4) by default, please modify source code to switch to other methods.")
    print("Enter to continue...")
    input()
    trade_s = 9
if neural_network.auto_switch == 1:
    pred_method = 4

#——————————————————导入测试数据——————————————————————
f=open("data\\"+test_file)
df=pd.read_csv(f)     #读入数据文件
data=df.iloc[:,index_size:(input_size+index_size+label+1)].values  #取第3-n列

#——————————————————准备报告数据——————————————————————
data_index0=df.iloc[test_begin-2:test_end-1,0].values
data_index1=df.iloc[test_begin-2:test_end-1,1].values
data_index2 = df.iloc[test_begin - 2:test_end - 1, 2].values
data_index3 = df.iloc[test_begin - 2:test_end - 1, 3].values
data_index4 = df.iloc[test_begin - 2:test_end - 1, 4].values
data_index5 = df.iloc[test_begin - 2:test_end - 1, 5].values
if pred_method==3: #测试方法3
    index0 = []
    index1 = []
    index2 = []
    index3 = []
    index4 = []
    index5 = []
    for i in range(len(data_index0) - time_step + 1):  #滚动取值
        i0 = data_index0[i:i + time_step]
        i1 = data_index1[i:i + time_step]
        i2 = data_index2[i:i + time_step]
        i3 = data_index3[i:i + time_step]
        i4 = data_index4[i:i + time_step]
        i5 = data_index5[i:i + time_step]
        index0.extend(i0.tolist())
        index1.extend(i1.tolist())
        index2.extend(i2.tolist())
        index3.extend(i3.tolist())
        index4.extend(i4.tolist())
        index5.extend(i5.tolist())
    data_index0 = index0
    data_index1 = index1
    data_index2 = index2
    data_index3 = index3
    data_index4 = index4
    data_index5 = index5
if pred_method == 0:
    data_index0 = data_index0[time_step-1:]
    data_index1 = data_index1[time_step-1:]
    data_index2 = data_index2[time_step-1:]
    data_index3 = data_index3[time_step-1:]
    data_index4 = data_index4[time_step-1:]
    data_index5 = data_index5[time_step-1:]

#——————————————————获取测试集——————————————————————
def get_test_data():
    test_x,test_y=[],[] #测试集
    data_test_x=data[test_begin-2:test_end-1,:input_size] #按配置文件，取测试features值，从n1行至n2行
    data_test_y=data[test_begin-2:test_end-1,input_size+label] #按配置文件，取测试Y值，从n1行至n2行
#——————————---自定义 计算标准化参数 的 数据和长度
    if nor_len_set == 0: #使用所有测试数据
        data_test_x_n=data[test_begin-2:test_end-1,:input_size]
        data_test_y_n=data[test_begin-2:test_end-1,input_size+label]
    if nor_len_set == 1: #使用所有训练数据
        data_test_x_n=data[train_begin-2:train_end-1,:input_size]
        data_test_y_n=data[train_begin-2:train_end-1,input_size+label]
    if nor_len_set == 2: #使用训练数据+测试数据
        data_test_x_n=data[train_begin-2:test_end-1,:input_size]
        data_test_y_n=data[train_begin-2:test_end-1,input_size+label]
    if nor_len_set == 3: #自定义长度
        data_test_x_n=data[train_end-1-(125-1):train_end-1,:input_size]
        data_test_y_n=data[train_end-1-(125-1):train_end-1,input_size+label]
#——————————---打印debug信息
    print("Debug information #1:")
    print("The len of original test data x:",len(data_test_x))
    print("The len of original test data y:",len(data_test_y))
    print("The first number in original test_x:",data_test_x[0])
    print("The last number in original test_x:",data_test_x[len(data_test_x)-1])
    print("The first number in original test_y:",data_test_y[0])
    print("The last number in original test_y:",data_test_y[len(data_test_y)-1])
#——————————--- X 标准化 和 计算Y 的标准化还原参数
    if nor_method == -1:
        normalized_test_data_x = data_test_x
        mean_y = 0
        std_y = 0
    if nor_method == 0:
        if nor_method_plus == 0:
            normalized_test_data_x=(data_test_x-np.mean(data_test_x_n,axis=0))/np.std(data_test_x_n,axis=0) #x标准化
            mean_y = np.mean(data_test_y_n, axis=0)  #y标准化还原参数
            std_y = np.std(data_test_y_n, axis=0)  #y标准化还原参数
        if nor_method_plus == 1:
            data_test_nor = data[test_begin - 2 - nor_shift_len + 1:test_end - 1, :input_size]
            if pred_method != 4:
                data_test_nor_y = data[test_begin - 2 - nor_shift_len - 4:test_end - 1, input_size + label]
            if pred_method == 4:
                data_test_nor_y = data[(test_begin - 2) + (time_step - 1) - (nor_shift_len + nextdays - 1):test_end - 1, input_size + label]
            normalized_test_data_x = []
            for i in range(len(data_test_x)):
                normalized_test_data_x0 = (data_test_x[i] - np.mean(data_test_nor[i:nor_shift_len + i], axis=0)) / np.std(data_test_nor[i:nor_shift_len + i], axis=0)
                normalized_test_data_x.append(normalized_test_data_x0)
            normalized_test_data_x = np.array(normalized_test_data_x)
            mean_y = []
            std_y = []
            if pred_method == 4:
                for i in range(len(data_test_y)-time_step+1):
                    mean_y0 = np.mean(data_test_nor_y[i:nor_shift_len + i], axis=0)
                    std_y0 = np.std(data_test_nor_y[i:nor_shift_len + i], axis=0)
                    mean_y.append(mean_y0)
                    std_y.append(std_y0)
            if pred_method != 4:
                for i in range(len(data_test_y)):
                    mean_y0 = np.mean(data_test_nor_y[i:nor_shift_len + i], axis=0)
                    std_y0 = np.std(data_test_nor_y[i:nor_shift_len + i], axis=0)
                    mean_y.append(mean_y0)
                    std_y.append(std_y0)
                if pred_method == 0:
                    mean_y = mean_y[time_step-1:]
                    std_y = std_y[time_step-1:]
                if pred_method == 3:
                    mean2 = []
                    std2 = []
                    for i in range(len(mean_y) - time_step + 1):
                        mean0 = mean_y[i:i + time_step]
                        std0 = std_y[i:i + time_step]
                        mean2.extend(mean0)
                        std2.extend(std0)
                    mean_y = mean2
                    std_y = std2
            print("The len of normalized test x:", len(normalized_test_data_x))
            print("The len of normalized mean_y:", len(mean_y))
            print("The len of normalized std_y:", len(std_y))
    if nor_method == 1:
        if nor_method_plus == 0:
            if mm_customize == 0:
                normalized_test_data_x = (data_test_x - np.min(data_test_x_n, axis=0)) / (np.max(data_test_x_n, axis=0) - np.min(data_test_x_n, axis=0))
                mean_y = np.min(data_test_y_n, axis=0)
                std_y = np.max(data_test_y_n, axis=0)
            if mm_customize == 1:
                normalized_test_data_x = (data_test_x - neural_network.mm_data_x[0]) / (neural_network.mm_data_x[1] - neural_network.mm_data_x[0])
                mean_y = neural_network.mm_data_y[0]
                std_y = neural_network.mm_data_y[1]
        if nor_method_plus == 1:
            data_test_nor = data[test_begin - 2 - nor_shift_len + 1:test_end - 1, :input_size]
            if pred_method != 4:
                data_test_nor_y = data[test_begin - 2 - nor_shift_len - 4:test_end - 1, input_size + label]
            if pred_method == 4:
                data_test_nor_y = data[(test_begin - 2) + (time_step - 1) - (nor_shift_len + nextdays - 1):test_end - 1, input_size + label]
            normalized_test_data_x = []
            for i in range(len(data_test_x)):
                normalized_test_data_x0 = (data_test_x[i] - np.min(data_test_nor[i:nor_shift_len + i], axis=0)) / (np.max(data_test_nor[i:nor_shift_len + i], axis=0) - np.min(data_test_nor[i:nor_shift_len + i], axis=0))
                normalized_test_data_x.append(normalized_test_data_x0)
            normalized_test_data_x = np.array(normalized_test_data_x)
            mean_y = []
            std_y = []
            if pred_method == 4:
                for i in range(len(data_test_y)-time_step+1):
                    mean_y0 = np.min(data_test_nor_y[i:nor_shift_len + i], axis=0)
                    std_y0 = np.max(data_test_nor_y[i:nor_shift_len + i], axis=0)
                    mean_y.append(mean_y0)
                    std_y.append(std_y0)
            if pred_method != 4:
                for i in range(len(data_test_y)):
                    mean_y0 = np.min(data_test_nor_y[i:nor_shift_len + i], axis=0)
                    std_y0 = np.max(data_test_nor_y[i:nor_shift_len + i], axis=0)
                    mean_y.append(mean_y0)
                    std_y.append(std_y0)
                if pred_method == 0:
                    mean_y = mean_y[time_step-1:]
                    std_y = std_y[time_step-1:]
                if pred_method == 3:
                    mean2 = []
                    std2 = []
                    for i in range(len(mean_y) - time_step + 1):
                        mean0 = mean_y[i:i + time_step]
                        std0 = std_y[i:i + time_step]
                        mean2.extend(mean0)
                        std2.extend(std0)
                    mean_y = mean2
                    std_y = std2
            print("The len of normalized test x:", len(normalized_test_data_x))
            print("The len of normalized mean_y:", len(mean_y))
            print("The len of normalized std_y:", len(std_y))
    print("mean_y:",mean_y)
    print("std_y:",std_y)
#——————————--- 构造 X 和 Y 测试数据
    if pred_method == 3:  # 测试方法3
        for i in range(len(normalized_test_data_x) - time_step + 1):  # 构造测试数据，X滚动取值
            x = normalized_test_data_x[i:i + time_step]
            test_x.append(x.tolist())
        for i in range(len(data_test_y) - time_step + 1):  # 构造验证数据，Y滚动取值
            test_y_0 = data_test_y[i:i + time_step]
            test_y.extend(test_y_0.tolist())
    if pred_method == 1:  #测试方法1
        size = (len(normalized_test_data_x) + time_step) // time_step  #有size-1个sample
        for i in range(size - 1): #构造测试数据，X切块取值，每块长为time_step
            x = normalized_test_data_x[i * time_step:(i + 1) * time_step]
            y = data_test_y[i * time_step:(i + 1) * time_step]
            test_x.append(x.tolist())
            test_y.extend(y) #Y实际为连续取值
    if pred_method == 0:
        for i in range(len(normalized_test_data_x)-time_step+1): #构造测试数据，X滚动取值
           x=normalized_test_data_x[i:i+time_step]
           test_x.append(x.tolist())
        test_y=data_test_y[time_step-1:] #Y从time_step开始连续取值
    if pred_method == 4:
        for i in range(len(normalized_test_data_x)-time_step+1): #构造测试数据，X滚动取值
           x=normalized_test_data_x[i:i+time_step]
           test_x.append(x.tolist())
        test_y=data_test_y[time_step-1:] #Y从time_step开始连续取值
    print("The len of final test data x:",len(test_x))
    print("The len of final test data y:",len(test_y))
    print("The first number in final test_y:",test_y[0])
    print("The last number in final test_y:", test_y[len(test_y)-1])
    return mean_y,std_y,test_x,test_y

#————————————————测试模型————————————————————
def predict(trade):
    trade_s = trade
    print("Test is being started now...")
    print("Normalization method used:",nor_method)
    if nor_method_plus == 0:
        print("Normalization plus is: off")
    if nor_method_plus == 1:
        print("Normalization plus is: on")
        print("Shift len set to:", nor_shift_len )
#——————————--- 获取测试数据 和 加载模型
    mean_y,std_y,test_x,test_y=get_test_data() #获取测试数据
    model=load_model("model\\"+str(train_method)+"_"+str(op)+"_"+model_input+"_T"+str(time_step)+"_label_"+str(label)+"_N"+str(nor_method)+"_model.h5") #读取保存的模型
    test_predict=[] #预测集
#——————————--- 预测
    if train_method == 0: #训练方法0，默认
        for step in range(len(test_x)):
            test_x_step = np.reshape(test_x[step], (1, time_step, input_size)) #调整X的维度为LSTM输入的要求
            prob = model.predict(test_x_step, batch_size=batch_size_cfg, verbose=1) #依据模型预测Y
            print("Input's sharp:", np.shape(test_x_step))
            print("Prob's sharp:", np.shape(prob))
            predict = prob.reshape((-1)) #调整预测Y的维度
            if (pred_method == 1) or (pred_method == 3):  #测试方法1 或者3
                test_predict.extend(predict) #把预测Y数组加入最终预测Y
            if pred_method == 0:  #测试方法0
                predict_last = [predict[(len(predict) - 1)]] #取预测Y数组的最后一个值
                test_predict.extend(predict_last) #加入最终预测Y
            if pred_method == 4:  #测试方法0
                test_predict.extend(predict)
#——————————--- 预测值 反标准化
    if nor_method == 0:
        if nor_method_plus == 0:
            test_predict=np.array(test_predict)*std_y+mean_y #预测Y反标准化
        if nor_method_plus == 1 and birthday != 1:
            test_predict_final = []
            for i in range(len(test_predict)):
                test_predict_f0 = test_predict[i] * std_y[i] + mean_y[i]
                test_predict_final.append(test_predict_f0)
            test_predict = np.array(test_predict_final)
        if nor_method_plus == 1 and birthday == 1:
            test_predict = np.array(test_predict)
    if nor_method == 1:
        if nor_method_plus == 0:
            test_predict = (np.array(test_predict) * (std_y - mean_y)) + mean_y
        if nor_method_plus == 1 and birthday != 1:
            test_predict_final = []
            for i in range(len(test_predict)):
                test_predict_f0 = test_predict[i] * (std_y[i] - mean_y[i]) + mean_y[i]
                test_predict_final.append(test_predict_f0)
            test_predict = np.array(test_predict_final)
        if nor_method_plus == 1 and birthday == 1:
            test_predict = np.array(test_predict)
#——————————--- 打印debug信息
    print("Debug information #2:")
    print("The len of prediction list:",len(test_predict))
    print("The len of actual data:",len(test_y))
    print("Prediction list:",test_predict)
    print("Actual data:",test_y)
#——————————--- 计算偏差程度
    acc=np.average(np.abs(test_predict - test_y[:len(test_predict)]) / np.abs(test_y[:len(test_predict)]))
#——————————--- 保存结果到文件
    f = open("data\\" + neural_network.config_file, "r")
    f1 = f.read()
    s0 = pd.Series(data_index0)
    s1 = pd.Series(data_index1)
    s2 = pd.Series(data_index2)
    s3 = pd.Series(data_index3)
    s4 = pd.Series(data_index4)
    s5 = pd.Series(data_index5)
    if pred_method == 0 or pred_method == 1 or pred_method ==3:
        test_y_s = test_y
        test_predict_s = test_predict
    if pred_method == 4:
        test_y_s, test_predict_s = [],[]
        temp = [0.0]
        for i in range(time_step-1):
            test_y_s.extend(temp)
            test_predict_s.extend(temp)
        test_y_s.extend(test_y)
        test_predict_s.extend(test_predict)
    s_y = pd.Series(test_y_s[:len(test_predict_s)])
    s_p = pd.Series(test_predict_s)
    s_cfg = pd.Series(f1)
    empty = np.full(len(test_predict_s),0.0)
    sr1 = pd.Series(empty)
    sr2 = pd.Series(empty)
    sr3 = pd.Series(empty)
    if nor_method_plus == 0:
        pr_c = ["Pred method:"+str(pred_method),"Nor method:"+str(nor_method),"Nor plus?: No","Nor len set:"+str(nor_len_set),"Accuracy:"+str(acc)]
    if nor_method_plus == 1:
        pr_c = ["Pred method:"+str(pred_method),"Nor method:"+str(nor_method),"Nor plus?: Yes","Nor shift len:"+str(nor_shift_len),"Accuracy:"+str(acc)]
    pr = pd.Series(pr_c)
    dr = pd.Series()
    current_time=str(time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time())))
    save = pd.DataFrame({"00-Cfg":s_cfg,"00-Index":s0,"01-Date":s1,"02-Open":s2,"03-Low":s3,"04-High":s4,"05-Close":s5,"06-Buy":sr1,"07-Sell":sr2,"08-Profit":sr3,"09-Actual":s_y,"10-Prediction":s_p,"11-Pred_Result":pr,"12-Deal":dr})
    if neural_network.auto_switch == 0:
        save.to_csv("model_result\\" + str(model_input) + "\\" + model_input + "_L" + str(label) + "_W" + str(pred_method) + "_T" + str(time_step) + "_N" + str(nor_method) + str(nor_method_plus) + "_OP" + str(op) + "_" + current_time + ".csv", index=False)
    if neural_network.auto_switch == 1:
        save.to_csv("auto_result\\" + str(model_input) + "\\" + model_input + "_L" + str(label) + "_W" + str(pred_method) + "_T" + str(time_step) + "_N" + str(nor_method) + str(nor_method_plus) + "_OP" + str(op) + "_" + current_time + ".csv", index=False)
#——————————--- 运行交易策略
    if trade_s != 9:
        add_result(trade,str(model_input) + "\\" + model_input + "_L" + str(label) + "_W" + str(pred_method) + "_T" + str(time_step) + "_N" + str(nor_method) + str(nor_method_plus) + "_OP" + str(op) + "_" + current_time + ".csv")
#——————————--- 打印结果
    print("### The test has been finished: ###")
    print("(0).Pred method:", pred_method, "|| Normalization method:", nor_method, "|| Y label no.:", label)
    print("(1).Data file:",test_file,"(from:",test_begin,"to:",test_end,")")
    print("(2).The latest five predictions:",test_predict[len(test_predict)-5:len(test_predict)])
    print("(3).The accuracy of this test:",acc)
    if neural_network.auto_switch == 0:
        print("(4).The result saved to:","model_result\\"+str(model_input) + "\\" + model_input + "_L" + str(label) + "_W" + str(pred_method) + "_T" + str(time_step) + "_N" + str(nor_method) + str(nor_method_plus) + "_OP" + str(op) + "_" + current_time + ".csv")
    if neural_network.auto_switch == 1:
        print("(4).The result saved to:","auto_result\\"+str(model_input) + "\\" + model_input + "_L" + str(label) + "_W" + str(pred_method) + "_T" + str(time_step) + "_N" + str(nor_method) + str(nor_method_plus) + "_OP" + str(op) + "_" + current_time + ".csv")
#——————————--- 以折线图表示结果
    plt.figure()
    plt.plot(list(range(len(test_predict))), test_predict, color='b')
    plt.plot(list(range(len(test_y))), test_y, color='r')
    plt.title('prediction vs actual data')
    plt.ylabel('value')
    plt.xlabel('time serial')
    plt.legend(['prediction','actual data'],loc='upper right')
    if neural_network.auto_switch == 0:
        plt.savefig("model_result\\"+str(model_input) + "\\" + model_input + "_L" + str(label) + "_W" + str(pred_method) + "_T" + str(time_step) + "_N" + str(nor_method) + str(nor_method_plus) + "_OP" + str(op) + "_" + current_time +".png", dpi=200)
    if neural_network.auto_switch == 1:
        plt.savefig("auto_result\\"+str(model_input) + "\\" + model_input + "_L" + str(label) + "_W" + str(pred_method) + "_T" + str(time_step) + "_N" + str(nor_method) + str(nor_method_plus) + "_OP" + str(op) + "_" + current_time +".png", dpi=200)
    if neural_network.auto_switch == 0:
        plt.show()

# ——————————————————交易策略应用——————————————————
def add_result(trade,result_file):
    if neural_network.auto_switch == 0:
        f_result=("model_result\\"+result_file)
    if neural_network.auto_switch == 1:
        f_result=("auto_result\\"+result_file)
    df=pd.read_csv(f_result) #读入结果文件
    open = df.iloc[:,3].values
    close = df.iloc[:,6].values
    buy = df.iloc[:,7].values
    sell = df.iloc[:,8].values
    profit = df.iloc[:,9].values
    actual = df.iloc[:,10].values
    predict = df.iloc[:,11].values
    deal = df.iloc[:,13].values
    size = int(len(predict)/time_step)
    pred_window = []
    print("Now, it is calculating buy/sell strategy...")
    win_num = 0
    loss_num = 0
    zero_num = 0
    strategy = trade
    if strategy == 5.2:
        for i in range(len(predict)-time_step-5):
            index = time_step-1+i
            pred_today = predict[index]
            close_today = close[index]
            open_tomorrow = open[index+1]
            if pred_today > open_tomorrow*1.02:
                buy[index] = open_tomorrow
                if actual[index] >= predict[index]:
                    sell[index] = predict[index]
                if actual[index] < predict[index]:
                    sell[index] = close[index+5]
                profit[index] = sell[index] - buy[index]
                if profit[index] > 0:
                    win_num = win_num + 1
                if profit[index] < 0:
                    loss_num = loss_num +1
                if profit[index] == 0:
                    zero_num = zero_num +1
            print("Round #",i)
            #print("Min of pred window:",pred_window.min())
            print("Last close:",close[index])
            print("Buy price:",buy[index])
            print("Sell price:",sell[index])
            print("Profit:",profit[index])
    if strategy == 5.1:
        for i in range(len(predict)-time_step):
            index = time_step-1+i
            pred_today = predict[index]
            close_today = close[index]
            open_tomorrow = open[index+1]
            if pred_today > open_tomorrow*1.02:
                buy[index] = open_tomorrow
                sell[index] = actual[index]
                profit[index] = sell[index] - buy[index]
                if profit[index] > 0:
                    win_num = win_num + 1
                if profit[index] < 0:
                    loss_num = loss_num +1
                if profit[index] == 0:
                    zero_num = zero_num +1
            print("Round #",i)
            #print("Min of pred window:",pred_window.min())
            print("Last close:",close[index])
            print("Buy price:",buy[index])
            print("Sell price:",sell[index])
            print("Profit:",profit[index])
    if strategy == 4:
        for i in range(size-1):
            i5 = int(i+(4*i)+4)
            pred_window = predict[i*time_step:(i+1)*time_step]
            #if (pred_window[0]<pred_window[2]) and (pred_window[1]<pred_window[3]) and (pred_window[2]<pred_window[4]) and (pred_window[0]<pred_window[3]):
            if pred_window[0]<pred_window[1]<pred_window[2]<pred_window[3]<pred_window[4]:
            #if (pred_window[0]<pred_window[1]<pred_window[2]) or (pred_window[1]<pred_window[2]<pred_window[3]) or (pred_window[2]<pred_window[3]<pred_window[4]):
            #if pred_window[0]<pred_window[1]<pred_window[2]<pred_window[3] or pred_window[1]<pred_window[2]<pred_window[3]<pred_window[4]:
                if pred_method == 1:
                    buy[i5] = open[i5+1]
                if pred_method == 3:
                    buy[i5] = open[i5+time_step]
                sell[i5] = actual[i5]
                profit[i5] = sell[i5] - buy[i5]
                if profit[i5] > 0:
                    win_num = win_num + 1
                if profit[i5] < 0:
                    loss_num = loss_num +1
                if profit[i5] == 0:
                    zero_num = zero_num +1
            print("Round #",i)
            print("Min of pred window:",pred_window.min())
            print("Last close:",close[i5])
            print("Buy price:",buy[i5])
            print("Sell price:",sell[i5])
            print("Profit:",profit[i5])
    if strategy == 3:
        for i in range(size-1):
            i5 = int(i+(4*i)+4)
            pred_window = predict[i*time_step:(i+1)*time_step]
            if (close[i5] < pred_window.min()) and (pred_window[0]<pred_window[2]) and (pred_window[1]<pred_window[3]) and (pred_window[2]<pred_window[4]) and (pred_window[0]<pred_window[3]):
                if pred_method == 1:
                    buy[i5] = open[i5+1]
                if pred_method == 3:
                    buy[i5] = open[i5+time_step]
                sell[i5] = actual[i5]
                profit[i5] = sell[i5] - buy[i5]
                if profit[i5] > 0:
                    win_num = win_num + 1
                if profit[i5] < 0:
                    loss_num = loss_num +1
                if profit[i5] == 0:
                    zero_num = zero_num +1
            print("Round #",i)
            print("Min of pred window:",pred_window.min())
            print("Last close:",close[i5])
            print("Buy price:",buy[i5])
            print("Sell price:",sell[i5])
            print("Profit:",profit[i5])
    if strategy == 2:
        for i in range(size-1):
            i5 = int(i+(4*i)+4)
            pred_window = predict[i*time_step:(i+1)*time_step]
            if (close[i5] < pred_window[4]) and (pred_window[0]<pred_window[2]) and (pred_window[1]<pred_window[3]) and (pred_window[2]<pred_window[4]) and (pred_window[0]<pred_window[3]):
                if pred_method == 1:
                    buy[i5] = open[i5+1]
                if pred_method == 3:
                    buy[i5] = open[i5+time_step]
                sell[i5] = actual[i5]
                profit[i5] = sell[i5] - buy[i5]
                if profit[i5] > 0:
                    win_num = win_num + 1
                if profit[i5] < 0:
                    loss_num = loss_num +1
                if profit[i5] == 0:
                    zero_num = zero_num +1
            print("Round #",i)
            print("Min of pred window:",pred_window.min())
            print("Last close:",close[i5])
            print("Buy price:",buy[i5])
            print("Sell price:",sell[i5])
            print("Profit:",profit[i5])
    if strategy == 1:
        for i in range(size-1):
            i5 = int(i+(4*i)+4)
            pred_window = predict[i*time_step:(i+1)*time_step]
            if close[i5] < pred_window.min():
                if pred_method == 1:
                    buy[i5] = open[i5+1]
                if pred_method == 3:
                    buy[i5] = open[i5+time_step]
                sell[i5] = actual[i5]
                profit[i5] = sell[i5] - buy[i5]
                if profit[i5] > 0:
                    win_num = win_num + 1
                if profit[i5] < 0:
                    loss_num = loss_num +1
                if profit[i5] == 0:
                    zero_num = zero_num +1
            print("Round #",i)
            print("Min of pred window:",pred_window.min())
            print("Last close:",close[i5])
            print("Buy price:",buy[i5])
            print("Sell price:",sell[i5])
            print("Profit:",profit[i5])
    deal_times = win_num + loss_num + zero_num
    ave_buy = np.sum(buy)/deal_times
    ave_sell = np.sum(sell)/deal_times
    total_pro = np.sum(profit)
    deal_s = ["Strategy:"+str(strategy),"Times:"+str(deal_times),"Total win:"+str(win_num),"Total loss:"+str(loss_num),"Total zero:"+str(zero_num),"Average buy price:"+str(ave_buy),"Average sell price:"+str(ave_sell),"Total profits per share:"+str(total_pro)]
    deal_status = []
    deal_status.extend(deal_s)
    deal_status.extend(deal[:len(deal)-8])
    df['12-Deal']=deal_status
    if neural_network.auto_switch == 0:
        df.to_csv("model_result\\"+result_file,index=False)
        print("Buy/sell strategy has been saved to: model_result\\"+result_file)
    if neural_network.auto_switch == 1:
        df.to_csv("auto_result\\"+result_file,index=False)
        print("Buy/sell strategy has been saved to: auto_result\\"+result_file)
    print("Good luck!")
# ——————————————————8848——————————————————
if neural_network.auto_switch == 0:
    predict(trade_s)