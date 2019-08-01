from keras.preprocessing import image
import numpy as np
import os
from configparser import ConfigParser
import tensorflow as tf
import torch
import cxr_dataset as CXR
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)# (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

if __name__ == "__main__":

    LABEL="Pneumonia"
    STARTER_IMAGES=True
    PATH_TO_IMAGES = "starter_images/"
    POSITIVE_FINDINGS_ONLY=True
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if not POSITIVE_FINDINGS_ONLY:
        finding = "any"
    else:
        finding = LABEL

    dataset = CXR.CXRDataset(
    path_to_images=PATH_TO_IMAGES,
    fold='test',
    transform=data_transform,
    finding=finding,
    starter_images=STARTER_IMAGES)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    inputs, labels, filename = next(iter(dataloader))
    dummy_input = torch.autograd.Variable(inputs.cpu())

    pb_model_path='experiments/3/checkpoint2.tf.pb'
    graph = tf.Graph()
    with graph.as_default():
        print("** load model **")
        graph = load_pb(pb_model_path)

    with tf.Session(graph=graph) as sess:
        output_tensor = graph.get_tensor_by_name('Sigmoid:0')
        input_tensor = graph.get_tensor_by_name('input:0')
        output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
        print(output)
