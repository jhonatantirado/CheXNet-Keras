from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from configparser import ConfigParser
from models.keras import ModelFactory
import tensorflow as tf

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img,data_format='channels_first')# (channels, height, width)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    return img_tensor

if __name__ == "__main__":

    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    output_dir = cp["DEFAULT"].get("output_dir")
    class_names = cp["DEFAULT"].get("class_names").split(",")

    pb_model = cp["TRAIN"].get("pb_model")
    pb_model_path = os.path.join(output_dir, pb_model)

    img_path_001 = 'data/images/00000344_000.PNG'
    new_image_001 = load_image(img_path_001)

    graph = tf.Graph()
    with graph.as_default():
        print("** load model **")
        graph = load_pb(pb_model_path)
        # print(graph.get_operations())

    with tf.Session(graph=graph) as sess:
        input = graph.get_tensor_by_name("input:0")
        feed_dict ={input:new_image_001}
        op_to_restore = graph.get_operation_by_name("Sigmoid")
        print (sess.run(op_to_restore,feed_dict))
