import os
import sys
sys.path.append('../')
from resnet import *
import torch

model = resnet34(num_classes=6, shortcut_type=True, sample_size=128, sample_duration=128)
weights = '../model/ct_pos_recogtion_20191115125435/ct_pos_recognition_0009_best.pth'
model.load_state_dict(torch.load(weights))

dummy_input = torch.randn(1,1,128,128,128)

torch.onnx.export(model, dummy_input, 'ctPosRecognition.onnx', verbose=True, input_names=['input'], output_names=['output'])

print('====> export to onnx model!')
