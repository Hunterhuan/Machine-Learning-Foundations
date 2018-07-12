import numpy as np
from numpy import random
import copy
import math

def load_data(filename):
    code = open(filename,"r")
    lines = code.readlines()
    xn = np.zeros((len(lines),3)).astype(np.float)
    yn = np.zeros((len(lines),)).astype(np.int)

    for i in range(len(lines)):
        line = lines[i]
        line = line.rstrip('\r\n').replace('\t',' ').split(' ')
        xn[i,0] = 1
        xn[i,1] = float(line[0])
        xn[i,2] = float(line[1])
        yn[i] = int(line[2])
    return xn,yn

def calculate_w_reg(x, y, lambda_value):
    return np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(), x) + lambda_value * np.eye(x.shape[1])), x.transpose()), y)

def calculate_E(w, x, y):
    scores = np.dot(w, x.T)
    E_out = 0
    for i in range(len(scores)):
        if scores[i]>=0 and y[i]!=1:
            E_out+=1
        elif scores[i]<0 and y[i]!=-1:
            E_out+=1
    return E_out/(1.0*len(scores))

def q13():
    train_x, train_y = load_data('hw4_train.dat')
    test_x, test_y = load_data('hw4_test.dat')
    lambda_value = 10
    W = calculate_w_reg(train_x, train_y, lambda_value)
    E_in = calculate_E(W, train_x,train_y)
    E_out = calculate_E(W, test_x, test_y)
    print(E_in)
    print(E_out)

def q14():
    train_x, train_y = load_data('hw4_train.dat')
    test_x, test_y = load_data('hw4_test.dat')
    E_in_min = 1000000
    E_out_min = 1000000
    lam = 0
    for i in range(-10,3):
        lambda_value = math.pow(10,i)
        W = calculate_w_reg(train_x, train_y, lambda_value)
        E_in = calculate_E(W, train_x,train_y)
        E_out = calculate_E(W, test_x, test_y)
        if E_in <= E_in_min:
            E_in_min = E_in
            E_out_min = E_out
            lam = i
    print(E_in_min)
    print(E_out_min)
    print(lam)

def q15():
    train_x, train_y = load_data('hw4_train.dat')
    test_x, test_y = load_data('hw4_test.dat')
    E_in_min = 1000000
    E_out_min = 1000000
    lam = 0
    for i in range(-10,3):
        lambda_value = math.pow(10,i)
        W = calculate_w_reg(train_x, train_y, lambda_value)
        E_in = calculate_E(W, train_x,train_y)
        E_out = calculate_E(W, test_x, test_y)
        if E_out <= E_out_min:
            E_in_min = E_in
            E_out_min = E_out
            lam = i
    print(E_in_min)
    print(E_out_min)
    print(lam)

def q16():
    train_x, train_y = load_data('hw4_train.dat')
    test_x, test_y = load_data('hw4_test.dat')
    train_x_1, train_y_1 = train_x[:120], train_y[:120]
    train_x_2, train_y_2 = train_x[120:], train_y[120:]
    E_train_min = 1000000
    E_val_min = 100000
    E_out_min = 1000000
    lam = 0
    for i in range(-10,3):
        lambda_value = math.pow(10,i)
        W = calculate_w_reg(train_x_1, train_y_1, lambda_value)
        E_train = calculate_E(W, train_x_1,train_y_1)
        E_val = calculate_E(W,train_x_2,train_y_2)
        E_out = calculate_E(W, test_x, test_y)
        if E_train <= E_train_min:
            E_train_min = E_train
            E_val_min = E_val
            E_out_min = E_out
            lam = i
    print(E_train_min)
    print(E_val_min)
    print(E_out_min)
    print(lam)
def q17():
    train_x, train_y = load_data('hw4_train.dat')
    test_x, test_y = load_data('hw4_test.dat')
    train_x_1, train_y_1 = train_x[:120], train_y[:120]
    train_x_2, train_y_2 = train_x[120:], train_y[120:]
    E_train_min = 1000000
    E_val_min = 100000
    E_out_min = 1000000
    lam = 0
    for i in range(-10,3):
        lambda_value = math.pow(10,i)
        W = calculate_w_reg(train_x_1, train_y_1, lambda_value)
        E_train = calculate_E(W, train_x_1,train_y_1)
        E_val = calculate_E(W,train_x_2,train_y_2)
        E_out = calculate_E(W, test_x, test_y)
        if E_val <= E_val_min:
            E_train_min = E_train
            E_val_min = E_val
            E_out_min = E_out
            lam = i
    print(E_train_min)
    print(E_val_min)
    print(E_out_min)
    print(lam)
def q18():
    train_x, train_y = load_data('hw4_train.dat')
    test_x, test_y = load_data('hw4_test.dat')
    lambda_value = 1
    W = calculate_w_reg(train_x, train_y, lambda_value)
    E_in = calculate_E(W, train_x,train_y)
    E_out = calculate_E(W, test_x, test_y)
    print(E_in)
    print(E_out)

def q19():
    train_x, train_y = load_data('hw4_train.dat')
    test_x, test_y = load_data('hw4_test.dat')
    folder_num = 5
    split_folder = 40
    Ecv_min = float("inf")
    optimal_lambda = 0
    for lambda_value in range(2, -11, -1):
        total_cv = 0
        for i in range(folder_num):
            # get test_data
            test_data_x = train_x[i * split_folder:(i + 1) * split_folder, :]
            test_data_y = train_y[i * split_folder:(i + 1) * split_folder]
            # train_data= raw_data-test_data，test_data可能在中间或两边
            if 0 < i < (folder_num - 1):
                train_data_x = np.concatenate((train_x[0:i * split_folder, :], train_x[(i + 1) * split_folder:, :]),
                                              axis=0)
                train_data_y = np.concatenate((train_y[0:i * split_folder], train_y[(i + 1) * split_folder:]), axis=0)
            elif i == 0:
                train_data_x = train_x[split_folder:, :]
                train_data_y = train_y[split_folder:]
            else:
                train_data_x = train_x[0:i * split_folder, :]
                train_data_y = train_y[0:i * split_folder]
            w_reg = calculate_w_reg(train_data_x, train_data_y, pow(10, lambda_value))
            Ecv = calculate_E(w_reg, test_data_x, test_data_y)
            total_cv += Ecv
        total_cv = total_cv * 1.0 / folder_num
        if Ecv_min > total_cv:
            Ecv_min = total_cv
            optimal_lambda = lambda_value
    print(optimal_lambda)
    print(Ecv_min)

def q20():
    optimal_lambda = -8
    train_x, train_y = load_data('hw4_train.dat')
    test_x, test_y = load_data('hw4_test.dat')
    w_reg = calculate_w_reg(train_x, train_y, pow(10, optimal_lambda))
    Ein = calculate_E(w_reg, train_x, train_y)
    Eout = calculate_E(w_reg, test_x, test_y)
    print(Ein)
    print(Eout)
if __name__=='__main__':
	q20()