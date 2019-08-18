from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from configparser import ConfigParser
from models.keras import ModelFactory
import tensorflow as tf


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":

    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

	# parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path)

    # image path
    img_path_001 = 'starter_images/00001698_000.PNG'
    img_path_002 = 'starter_images/00003728_000.PNG'
    img_path_003 = 'starter_images/00005318_000.PNG'
    # load a single image
    new_image_001 = load_image(img_path_001)
    new_image_002 = load_image(img_path_002)
    new_image_003 = load_image(img_path_003)
    # check prediction
    pred_001 = model.predict(new_image_001)
    pred_002 = model.predict(new_image_002)
    pred_003 = model.predict(new_image_003)

    print (pred_001)
    print (pred_002)
    print (pred_003)

    result_001 = tf.argmax(pred_001, 1)
    result_002 = tf.argmax(pred_002, 1)
    result_003 = tf.argmax(pred_003, 1)

    predicted_class_001 = tf.keras.backend.eval(result_001)
    predicted_class_002 = tf.keras.backend.eval(result_002)
    predicted_class_003 = tf.keras.backend.eval(result_003)

    print (predicted_class_001)
    print (predicted_class_002)
    print (predicted_class_003)

    print (class_names[predicted_class_001[0]])
    print (class_names[predicted_class_002[0]])
    print (class_names[predicted_class_003[0]])
