 #!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2022 2022/3/28 9:43 下午
@Author  : keyoung
@File    : NeuralNetwork_demo3.py
"""
import random
import matplotlib.pyplot as plt
import numpy as np
class applenetwork:
    def __init__(self,inputnode,hidenode,outputnode):
         # self.param = random.random()
         self.inodes = inputnode
         self.hnodes = hidenode
         self.onodes = outputnode
         self.w_in_hide = np.random.rand(self.hnodes,self.inodes)       #输入与隐藏之间的权值
         self.w_out_hide = np.random.rand(self.onodes,self.hnodes)      #隐藏与输出之间的权值
         self.lr = 0.01
        #激活函数
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def softmax(self,a):
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a/sum_exp_a

    def train(self,input_data,output_data):
     temp_re = self.forward(input_data)
     err = output_data - temp_re
     diff = err / input_data
     self.param = self.param + diff*self.lr

    def loss(self,test_data):
     total_loss = 0
     for item in test_data:
         result = self.forward(item[0])
         err = abs(item[1] - result )
         total_loss = total_loss + err
     return total_loss

    def query(self,w):
     # return self.param * w
     return self.forward(w)

    def forward(self,input_data):
        hidden_input = np.dot(self.w_in_hide,input_data)
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(self.w_out_hide,hidden_output)
        return final_input

if __name__ == '__main__':
    n = applenetwork()
    x = []
    y = []
    train_data = [[2,4],[1,2],[1.5,2.9],[5,9.9],[3,6.2],[4,7],[3,6.1]]
    test_data = [[2,4],[5,10.1],[7,13.9]]
    #训练10次
    step = 0
    for epoch in range(0,50):  #训练次数
     for i in train_data:
         n.train(i[0],i[1])
         err =n.loss(test_data)
         step += 1
         x.append(step)
         y.append(err)

    # 训练完以后查询
    print(n.query(3))
    print(n.query(8))

    plt.figure(num=1,figsize=(5,5))
    plt.scatter(x,y)
    plt.show()