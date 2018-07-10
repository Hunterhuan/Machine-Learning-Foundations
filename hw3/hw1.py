import numpy as np
from numpy import random
import copy

def target_func(x1,x2):
    return (1 if(x1*x1 + x2*x2 -0.6) >=0 else -1)

def train_data_with_random_error(num = 1000):
    features = np.zeros((num,3))
    labels = np.zeros((num,1))
    points_x1 = random.uniform(-1,1,num)
    points_x2 = random.uniform(-1,1,num)
    for i in range(num):
        features[i,0] = 1
        features[i,1] = points_x1[i]
        features[i,2] = points_x2[i]
        labels[i] = target_func(points_x1[i],points_x2[i])
        if i<= num*0.1:
            labels[i] *= -1
    return features,labels
def feature_transform(features):
    new = np.zeros((len(features),6))
    new[:,0:3] = features[:,:]*1
    new[:,3] = features[:,1] * features[:,2]
    new[:,4] = features[:,1] * features[:,1]
    new[:,5] = features[:,2] * features[:,2]
    return new

def error_rate(features, labels,w):
    wrong = 0
    for i in range(len(labels)):
        if np.dot(features[i],w)*labels[i,0] <0:
            wrong+=1
    return wrong/(len(labels)*1.0)

def linear_regression_closed_form(X,Y):
    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)

def load_data(file_path):
    f = open(file_path)
    try:
        lines = f.readlines()
    finally:
        f.close()
    example_num = len(lines)
    feature_dimension = len(lines[0].strip().split())
    features = np.zeros((example_num,feature_dimension))
    features[:,0] = 1
    labels = np.zeros((example_num,1))
    for index,line in enumerate(lines):
        items = line.strip().split(' ')
        features[index,1:] = [float(str_num) for str_num in items[0:-1]]
        labels[index] = float(items[-1])
    return features, labels

def gradient_descent(X, y ,w):
    tmp = -y * (np.dot(X,w))
    weight_matrix = np.exp(tmp)/((1+np.exp(tmp))*1.0)
    gradient = 1/(len(X) * 1.0) * (sum(weight_matrix* -y * X).reshape(len(w),1))
    return gradient

def stochastic_gradient_descent(X, y ,w):
    tmp = -y * (np.dot(X,w))
    weight = np.exp(tmp)/((1 + np.exp(tmp)) * 1.0)
    gradient = weight * -y * X
    return gradient.reshape(len(gradient), 1)

class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X, y, Eta = 0.001, max_iteration = 2000, sgd = False):
        self.__w = np.zeros((len(X[0]),1))
        if not sgd:
            for i in range(max_iteration):
                self.__w = self.__w - Eta * gradient_descent(X,y,self.__w)
        else:
            index = 0
            for i in range(max_iteration):
                if(index >= len(X)):
                    index = 0
                self.__w = self.__w - Eta * stochastic_gradient_descent(np.array(X[index]), y[index], self.__w)
                index += 1

    def predict(self, X):
        binary_result = np.dot(X, self.__w) >= 0
        return np.array([(1 if _ >0 else -1) for _ in binary_result]).reshape(len(X), 1)

    def get_w(self):
        return self.__w

    def score(self, X, y):
        predict_y = self.predict(X)
        return sum(predict_y != y)/(len(y)*1.0)





def q13():
    error = 0*1.0
    for i in range(1000):
        features, labels = train_data_with_random_error(1000)
        w = linear_regression_closed_form(features,labels)
        error += error_rate(features, labels, w)
    print(error/(1000*1.0))
def q14():
    res = np.zeros((6,6))
    min_error_in = float("inf")
    for i in range(1000):
        features, labels = train_data_with_random_error(1000)
        new_features = feature_transform(features)
        w = linear_regression_closed_form(new_features,labels)
        error_in = error_rate(new_features,labels,w)
        if error_in <= min_error_in:
            res = w
            min_error_in = error_in
    print(res)
def q15():
    w = np.array([-1,-0.05,0.08,0.13,1.5,1.5])
    error_out = 0
    for i in range(1000):
        features, labels = train_data_with_random_error(1000)
        new_features = feature_transform(features)
        error_out += error_rate(new_features,labels,w)
    print(error_out/1.0/1000)
def q18():
    X, Y = load_data("hw3_train.dat")
    lr = LinearRegression()
    lr.fit(X, Y ,max_iteration = 2000)
    test_X, test_Y = load_data("hw3_test.dat")
    print(lr.score(test_X,test_Y))
def q19():
    X, Y = load_data("hw3_train.dat")
    lr = LinearRegression()
    lr.fit(X, Y, 0.01, 2000)
    test_X, test_Y = load_data("hw3_test.dat")
    print(lr.score(test_X,test_Y))

def q20():
    X, Y = load_data("hw3_train.dat")
    lr = LinearRegression()
    lr.fit(X, Y, sgd = True)
    test_X, test_Y = load_data("hw3_test.dat")
    print(lr.score(test_X,test_Y))
if __name__=='__main__':
	#problem_17_18()
    q20()