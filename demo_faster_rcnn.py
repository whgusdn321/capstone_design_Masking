"""02. Predict with pre-trained Faster RCNN models
==============================================

This article shows how to play with pre-trained Faster RCNN model.

First let's import some necessary libraries:
"""

from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
import os
import mxnet as mx

ctx = [mx.gpu(0), mx.gpu(1)]
model_PATH = r'C:\Users\suer0426\PycharmProjects\Project_X\Pufaster_rcnn_resnet50_v1b_custom_best.params'
net1 = model_zoo.get_model('faster_rcnn_resnet50_v1b_custom', classes = 'face', pretrained=False, ctx = ctx[0])
net1.load_parameters(model_PATH, ctx =ctx[0])
print('net1 loaded successfully ')

model_PATH2 = r"C:\Users\suer0426\Desktop\Desktop (2)\new_car_faster_rcnn_resnet50_v1b_custom_best.params"
net2 = model_zoo.get_model('faster_rcnn_resnet50_v1b_custom', classes = ['car'], pretrained=False, ctx = ctx[1])
net2.load_parameters(model_PATH2, ctx =ctx[1])
print('whole net loaded sucessfully!')

