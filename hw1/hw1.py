import numpy as np
import random
import copy

def read_data(path):
    f1=open(path)
    x_matrix=[]
    y_matrix=[]
    for i in f1:
        x=[1]
        for j in i.split('\t')[0].split():
            x.append(float(j))
        y_matrix.append(int(i.strip().split('\t')[1]))
        x_matrix.append(x)
    x_matrix=np.array(x_matrix)
    #print(x_matrix)
    return x_matrix,y_matrix


def sign(x,w):
    if np.dot(x, w)[0]<=0:
        return -1
    else:
        return 1

def naive_PLA(x_matrix,y_matrix):
    sum=len(x_matrix)
    length=len(x_matrix[0])
    w=np.zeros((length,1))
    #print(w)
    count=0
    s=0
    flag=0
    while True:
        for i in range(sum):
            s+=1
            #print(np.dot(x_matrix[i], w)[0]*y_matrix[i])
            if sign(x_matrix[i], w)!=y_matrix[i]:
                #print(w,x_matrix[i],y_matrix[i])
                w+=np.matrix(x_matrix[i]).T*y_matrix[i]  # w =  w + x*y
                count+=1
                s=0
            if s==sum:
                flag=1
                break
        if flag==1:
            break
    return count

def random_PLA1(x_matrix,y_matrix):
    sum=len(x_matrix)
    length=len(x_matrix[0])
    w=np.zeros((length,1))
    order=range(sum)
    #print(order)
    random_seed=random.sample(order,sum)
    #print(random_seed)
    count=0
    s=0
    flag=0
    while True:
        for i in random_seed:
            s+=1
            #print(np.dot(x_matrix[i], w)[0]*y_matrix[i])
            if sign(x_matrix[i], w)!=y_matrix[i]:
                #print(w,x_matrix[i],y_matrix[i])
                w+=np.matrix(x_matrix[i]).T*y_matrix[i]
                count+=1
                s=0
            if s==sum:
                flag=1
                break
        if flag==1:
            break
    return count
 
def weighted_random_PLA(x_matrix,y_matrix,eta):
    sum=len(x_matrix)
    length=len(x_matrix[0])
    w=np.zeros((length,1))
    order=range(sum)
    #print(order)
    random_seed=random.sample(order,sum)
    #print(random_seed)
    count=0
    s=0
    flag=0
    while True:
        for i in random_seed:
            s+=1
            #print(np.dot(x_matrix[i], w)[0]*y_matrix[i])
            if sign(x_matrix[i], w)!=y_matrix[i]:
                #print(w,x_matrix[i],y_matrix[i])
                w+=np.matrix(x_matrix[i]).T*y_matrix[i]*eta
                count+=1
                s=0
            if s==sum:
                flag=1
                break
        if flag==1:
            break
    return count
def test(w,x_matrix,y_matrix,sum):
    count = 0
    for i in range(sum):
        if sign(x_matrix[i],w) != y_matrix[i]:
            count +=1
    return count

def random_pocket(x_matrix,y_matrix,updates):
    sum=len(x_matrix)
    length=len(x_matrix[0])
    w=np.zeros((length,1))
    order=range(sum)
    #print(order)
    random_seed=random.sample(order,sum)
    bestw = np.zeros((length,1))
    bestCount = 501
    update = 0
    while update<updates:
        for i in random_seed:
            if sign(x_matrix[i],w) != y_matrix[i]:
                update +=1
                w = w + np.matrix(x_matrix[i]).T * y_matrix[i]
                count = test(w,x_matrix,y_matrix,sum)
                if count<bestCount:
                    bestCount = count
                    bestw = w
            if update==updates:
                break
    return bestw

def random_PLA2(x_matrix,y_matrix,updates):
    sum=len(x_matrix)
    length=len(x_matrix[0])
    w=np.zeros((length,1))
    order=range(sum)
    #print(order)
    random_seed=random.sample(order,sum)
    update = 0
    s=0
    while update < updates:
        for i in random_seed:
            s+=1
            #print(np.dot(x_matrix[i], w)[0]*y_matrix[i])
            if sign(x_matrix[i], w)!=y_matrix[i]:
                #print(w,x_matrix[i],y_matrix[i])
                w+=np.matrix(x_matrix[i]).T*y_matrix[i]
                update+=1
                s=0
            if update==updates:
                break
            if s==sum:
                flag=1
                break
    return w

def problem_15():
    x_matrix, y_matrix=read_data('hw1_15_train.dat')
    count=naive_PLA(x_matrix,y_matrix)       #Question 15
    print(count)
    return
def problem_16():
    x_matrix, y_matrix=read_data('hw1_15_train.dat')
    sum=0                                     #Question 16
    for i in range(2000):
        ppp = random_PLA1(x_matrix,y_matrix)
        #print(ppp)
        sum += ppp;
    print(sum/2000)
    return
def problem_17():
    x_matrix, y_matrix=read_data('hw1_15_train.dat')
    sum=0                                     #Question 17
    for i in range(2000):
        sum+=weighted_random_PLA(x_matrix,y_matrix,0.5)
    print(sum/2000)
    return
def problem_18():
    x_matrix, y_matrix=read_data('hw1_18_train.dat')
    x_test, y_test = read_data('hw1_18_test.dat')
    sum = len(x_test)
    error = 0
    time = 2000
    for i in range(time):
        w = random_pocket(x_matrix,y_matrix,50)
        count = test(w,x_test,y_test,sum)
        error +=count
    print(float(error)/sum/time)
    return

def problem_19():
    x_matrix, y_matrix=read_data('hw1_18_train.dat')
    x_test, y_test = read_data('hw1_18_test.dat')
    sum = len(x_test)
    error = 0
    for i in range(2000):
        w = random_PLA2(x_matrix,y_matrix,50)
        error += test(w,x_test,y_test,sum)
    print(float(error)/sum/2000)
    return
def problem_20():
    x_matrix, y_matrix=read_data('hw1_18_train.dat')
    x_test, y_test = read_data('hw1_18_test.dat')
    sum = len(x_test)
    error = 0
    time = 300
    for i in range(time):
        w = random_pocket(x_matrix,y_matrix,100)
        count = test(w,x_test,y_test,sum)
        error +=count
    print(float(error)/sum/time)
    return
if __name__=='__main__':
	problem_15()
    problem_20()