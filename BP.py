import math
import random

def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):  # 创造一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivate(x):
    return x * (1 - x)

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
        for i in range(self.num_hidden2):
            res = 0.0
            for j in range(self.num_hidden1):
                res += self.hidden1_cells[j] * self.hidden_weights[j][i]
            self.hidden2_cells[i] = sigmoid(res)
        for i in range(self.num_output):
            res = 0.0
            for j in range(self.num_hidden2):
                res += self.hidden2_cells[j] * self.output_weights[j][i]
            tmp = sigmoid(res)
            self.output_cells[i] = 1 if tmp > 0.5 else 0

    
    def backPropagation(self, case, label, learning_rate, correct_rate):
        delta_output = [0.0] * self.num_output
        for i in range(self.num_output):
            error = label[i] - self.output_cells[i]
            delta_output[i] = sigmoid_derivate(self.output_cells[0]) * error
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
                self.output_correction[i][j] += change
        for i in range(self.num_hidden1):
            for j in range(self.num_hidden2):
                change = delta_hidden2[j] * self.hidden1_cells[i]
                self.hidden_weights[i][j] += learning_rate * change + correct_rate * self.hidden_correction[i][j]
                self.hidden_correction[i][j] += change
        for i in range(self.num_input):
            for j in range(self.num_hidden1):
                change = delta_hidden1[j] * self.input_cells[i]
                self.input_weights[i][j] += learning_rate * change + correct_rate * self.input_correction[i][j]
                self.input_correction[i][j] += change
        global_error = 0.0
        for i in range(len(label)):
            global_error += 0.5 * (label[i] - self.output_cells[i]) ** 2
        return global_error

    def train(self, cases, labels, max_epochs = 10000, learning_rate = 0.05, correct_rate = 0.1):
        
        for epoch in range(max_epochs):
            error = 0.0
            for i, case in enumerate(cases):
                label = labels[i]
                self.predict(case)
                error += self.backPropagation(case, label, learning_rate, correct_rate)

    def test(self):
        pass

if __name__ == "__main__":
    pass