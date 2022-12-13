import math
import random
import numpy as np
import pandas as pd
import scipy.special

def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):  # 创造一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat

def relu(input):
    return np.maximum(0, input)

def sigmoid(x):
    try:
        result = 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        result = math.inf
    return result

def sigmoid_derivate(x):
    return x * (1 - x)

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def softmax_derivate(x, eta):
    dout = np.diag(x) - np.outer(x, x)
    return np.dot(dout, eta)

class BPNeuralNet:
    """BP神经网络(包含两层隐藏层)"""
    
    def __init__(self):
        """
        input_weights:输入层与第一层隐藏层之间的权值矩阵
        hidden_weights:第一层隐藏层与第二层隐藏层之间的权值矩阵
        output_weights:第二层隐藏层与输出层之间的权值矩阵
        
        output_cells: 输出层的输出
        hidden2_cells: 第二个隐藏层的输出
        hidden1_cells: 第一个隐藏层的输出
        input_cells: 输入层的输出
        """
        self.num_input = 0
        self.num_hidden1 = 0
        self.num_hidden2 = 0
        self.num_output = 0
        self.input_cells = []
        self.input_weights = []
        self.input_correction = []
        self.hidden1_cells = []
        self.hidden2_cells = []
        self.hidden_weights = []
        self.hidden_correction = []
        self.output_cells = []
        self.output_weights = []
        self.output_correction = []
        
 
    def setup(self, num_input, num_hidden1, num_hidden2, num_output):
        """
        num_input:输入层节点个数
        num_hidden1:第一个隐藏层节点个数
        num_hidden2:第二个隐藏层节点个数
        num_output:输出层节点个数
        """
        self.num_input = num_input + 1
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_output = num_output

        self.input_cells = [1.0] * self.num_input
        self.hidden1_cells = [1.0] * self.num_hidden1
        self.hidden2_cells = [1.0] * self.num_hidden2
        self.output_cells = [1.0] * self.num_output

        self.input_weights = make_matrix(self.num_input, self.num_hidden1)
        self.hidden_weights = make_matrix(self.num_hidden1, self.num_hidden2)
        self.output_weights = make_matrix(self.num_hidden2, self.num_output)

        #init weight
        for i in range(self.num_input):
            for j in range(self.num_hidden1):
                self.input_weights[i][j] = rand(-0.2, 0.2)
        for i in range(self.num_hidden1):
            for j in range(self.num_hidden2):
                self.hidden_weights[i][j] = rand(-0.2, 0.2)
        for i in range(self.num_hidden2):
            for j in range(self.num_output):
                self.output_weights[i][j] = rand(-0.2, 0.2)

        self.input_correction = make_matrix(self.num_input, self.num_hidden1)
        self.hidden_correction = make_matrix(self.num_hidden1, self.num_hidden2)
        self.output_correction = make_matrix(self.num_hidden2, self.num_output)
 

    def predict(self, inputs):
        for i in range(self.num_input - 1):
            self.input_cells[i] = inputs[i]
        for i in range(self.num_hidden1):
            res = 0.0
            for j in range(self.num_input):
                res += self.input_cells[j] * self.input_weights[j][i]
            self.hidden1_cells[i] = sigmoid(res)
            # self.hidden1_cells[i] = relu(res)
        for i in range(self.num_hidden2):
            res = 0.0
            for j in range(self.num_hidden1):
                res += self.hidden1_cells[j] * self.hidden_weights[j][i]
            self.hidden2_cells[i] = sigmoid(res)
            # self.hidden2_cells[i] = relu(res)
        for i in range(self.num_output):
            res = 0.0
            for j in range(self.num_hidden2):
                res += self.hidden2_cells[j] * self.output_weights[j][i]
            self.output_cells[i] = res
        self.output_cells = softmax(self.output_cells)
        # self.output_cells = scipy.special.softmax(self.output_cells, axis=0)
        return int(np.argmax(self.output_cells))

    
    def backPropagation(self, case, label, learning_rate, correct_rate):
        pred = self.predict(case)
        delta_output = [0.0] * self.num_output
        error = label - self.output_cells
        delta_output = softmax_derivate(self.output_cells, error)
        
        delta_hidden2 = [0.0] * self.num_hidden2
        for i in range(self.num_hidden2):
            error = 0.0
            for j in range(self.num_output):
                error += delta_output[j] * self.output_weights[i][j]
            delta_hidden2[i] = sigmoid_derivate(self.hidden2_cells[i]) * error
        delta_hidden1 = [0.0] * self.num_hidden1
        for i in range(self.num_hidden1):
            error = 0.0
            for j in range(self.num_hidden2):
                error += delta_hidden2[j] * self.hidden_weights[i][j]
            delta_hidden1[i] = sigmoid_derivate(self.hidden1_cells[i]) * error
        #更新权重
        for i in range(self.num_hidden2):
            for j in range(self.num_output):
                change = delta_output[j] * self.hidden2_cells[i]
                self.output_weights[i][j] += learning_rate * change + correct_rate * self.output_correction[i][j]
                self.output_correction[i][j] = change
        for i in range(self.num_hidden1):
            for j in range(self.num_hidden2):
                change = delta_hidden2[j] * self.hidden1_cells[i]
                self.hidden_weights[i][j] += learning_rate * change + correct_rate * self.hidden_correction[i][j]
                self.hidden_correction[i][j] = change
        for i in range(self.num_input):
            for j in range(self.num_hidden1):
                change = delta_hidden1[j] * self.input_cells[i]
                self.input_weights[i][j] += learning_rate * change + correct_rate * self.input_correction[i][j]
                self.input_correction[i][j] = change
        global_error = 0.0  
        for i in range(len(label)):
            global_error += 0.5 * (label[i] - self.output_cells[i]) ** 2
        return global_error

    def train(self, cases, labels, max_epochs = 1000, learning_rate = 50, correct_rate = 0.98):
        for epoch in range(max_epochs):
            error = 0.0
            for i, case in enumerate(cases):
                label = labels[i]
                error += self.backPropagation(case, label, learning_rate, correct_rate)
            if epoch % 20 == 0: #每20轮显示一次
                print(f'Epoch {epoch}, error : {error}')

    def test(self, test_data, test_label):
        true = 0
        for i in range(len(test_data)):
            pred = self.predict(test_data[i])
            ground_truth = int(np.argmax(test_label[i]))
            if ground_truth == pred:
                true += 1
        print(f'Test accuracy : {true / len(test_data)}')


if __name__ == "__main__":
    #数据集加载
    df = pd.read_csv('./data/iris.data', delimiter=',', header=None)
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # print(df.shape) #(150, 5)

    #对数据进行预处理
    labels_mapping = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    df['label'] = df['label'].map(labels_mapping)
    data = np.array(df)
    X, y = data[:,:-1], data[:,-1]
    # print(X.shape, y.shape)  #(150, 4) (150,)
    
    #划分训练集、测试集
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.2, random_state=666)
    # print(train_data.shape, test_data.shape, train_label.shape, test_label.shape) #(120, 4) (30, 4) (120,) (30,)

    #one-hot encoding
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(train_label)
    train_label = lb.transform(train_label)
    test_label = lb.transform(test_label)
    # print(train_label.shape, test_label.shape)  #(120, 3) (30, 3)
    
    #创建模型
    model = BPNeuralNet()
    model.setup(4, 20, 10, 3)

    #训练模型
    model.train(cases=train_data, labels=train_label, max_epochs=1000)

    #测试模型
    model.test(test_data, test_label)