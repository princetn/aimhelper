import tensorflow as tf
import tf_keras as k3
from tensorflow.python.framework.convert_to_constants import     convert_variables_to_constants_v2
import numpy as np
#Step 2: Load your TensorFlow model
model = k3.models.load_model(r'C:\Users\argas\Downloads\faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8\faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8\saved_model')

#Step 3: Convert the model to a concrete function
full_model = tf.function(lambda x: model(x))
# full_model = full_model.get_concrete_function(
#     tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.signatures['serving_default'].inputs[0].shape.as_list(), model.signatures['serving_default'].inputs[0].dtype.name))
#Step 4: Freeze the graph
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)




# frozen_func = tf.compat.v1.graph_util.convert_variables_to_constants(
#     full_model.graph.as_graph_def(),
#     [node.name for node in full_model.graph.get_operations()
#      if node.op == 'Identity']
# )


#Step 5: Save the frozen graph
tf.io.write_graph(
    graph_or_graph_def=frozen_func.graph,
    logdir='./frozen_models',
    name='frozen_graph.pb',
    as_text=False
)


