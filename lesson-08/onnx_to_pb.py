import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('ctPosRecognition.onnx')
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
onnx.save(onnx_model, 'dynamic_model.onnx')

# 如果在jupyter或ipython中运行，需要通过CUDA_VISIBLE_DEVICES=0或os.environ['CUDA_VISIBLE_DEVICES']='0'，指定一个gpu
# 否则会报xla相关的错误
tf_rep = prepare(onnx_model)

print(tf_rep.inputs)
print(tf_rep.outputs)
print(tf_rep.tensor_dict)

print('\n====> inputs:')
for inp in tf_rep.inputs:
    print(tf_rep.tensor_dict[inp])

print('\n====> outputs:')
for outp in tf_rep.outputs:
    print(tf_rep.tensor_dict[outp])

tf_rep.export_graph('ctPosRecognition.pb')
print('export to frozen pb!')
