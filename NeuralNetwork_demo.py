#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2022 2022/3/23 8:23 下午
@Author  : keyoung
@File    : NeuralNetwork_demo.py
"""
import random

# 神经网络简单例子认识
# example: buy apple
# 4块钱能买2斤苹果 问1块钱能买（ ）斤苹果？

class applenetwork:
    def __init__(self):
        #首先随机生产一个输入的价格
        self.param = random.random()

    def train(self,input_data,output_data): #质量，以及总价格 相当于input output
        #我们知道 param = wight/total
        temp_re = self.forward(input_data) #这里的中间结果一定要与下面的查询一致 表示前向传播
        err = output_data - temp_re

        # y = ax 随机生成一个斜率a
        # total = （a+&a）x  total 表示真正的值
        # &a = （total - y）/x

        diff = err / input_data
        self.param = self.param + diff

    def query(self,w):
        # return self.param * w
        return self.forward(w)

    def forward(self,w):
        return self.param * w

if __name__ == '__main__':
    n = applenetwork()
    n.train(4, 2)
    # 训练完以后查询
    print(n.query(3))