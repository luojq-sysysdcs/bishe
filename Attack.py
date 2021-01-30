#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/10 21:37
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : Attack.py
# @Software: PyCharm
class Attack():
    def __init__(self, name, model):
        self.attach = name
        self.model = model
        self._targeted = -1  # targeted attach(-1) or merely misclassify(1)
        self.device = next(model.parameters()).device

    def _transform_label(self, images, labels):
        return labels