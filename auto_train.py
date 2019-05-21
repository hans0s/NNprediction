#coding=utf-8

f_s = open("auto_cfg\\auto_switch.txt","w")
f_s.write("1")
f_s.close()

import neural_network
import ds_train
import numpy as np
import pandas as pd
import importlib

def menu():
    print()
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$         Auto Train        $$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("<Auto status:", neural_network.auto_switch, ">")
    print("Menu:")
    print("(1). Auto train now")
    print("(2). Show auto list")
    print("(3). Switch auto status")
    print("(4). Exit")
    print("Please select:")

def run():
    exit=1
    while exit==1:
        menu()
        select=int(input())
        if select == 1:
            for i in range(neural_network.auto_list_size):
                importlib.reload(neural_network)
                importlib.reload(ds_train)
                ds_train.train_nn()
                f_i = open("auto_cfg\\auto_index.txt","w")
                f_i.write(str(i+1))
                f_i.close()
            f_i = open("auto_cfg\\auto_index.txt","w")
            f_i.write("0")
            f_i.close()
        if select == 2:
            f_o = open("auto_cfg\\auto_list.txt","r")
            print("Showing auto list now...")
            print("Auto list:")
            print(f_o.read())
        if select == 3:
            print("to be added...")
        if select == 4:
            exit=0
            f_s = open("auto_cfg\\auto_switch.txt","w")
            f_s.write("0")
            f_s.close()
    print("Bye-bye!")
run()