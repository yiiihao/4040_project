import tensorflow as tf
import numpy as np
 
 
 
def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)
    return inputs
 
def red_tf(imgs,net_size):
    raw_image_dataset = tf.data.TFRecordDataset(imgs).shuffle(1000)
 
    image_feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/label': tf.io.FixedLenFeature([], tf.int64),
        'image/roi': tf.io.FixedLenFeature([4], tf.float32),
        'image/landmark': tf.io.FixedLenFeature([10],tf.float32)
    }
    def _parse_image_function(example_proto):
      # Parse the input tf.Example proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, image_feature_description)
 
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    #print(parsed_image_dataset)
    image_batch = []
    label_batch = []
    bbox_batch = []
    landmark_batch = []
 
    for image_features in parsed_image_dataset:
 
        image_raw = tf.io.decode_raw(image_features['image/encoded'],tf.uint8)
        # 将值规划在[-1,1]内
        images = tf.reshape(image_raw, [net_size, net_size, 3])
        image = (tf.cast(images, tf.float32) - 127.5) / 128
        #图像变色
        image = image_color_distort(image)
        image_batch.append(image)
 
        label = tf.cast(image_features['image/label'], tf.float32)
        label_batch.append(label)
 
        roi = tf.cast(image_features['image/roi'], tf.float32)
        bbox_batch.append(roi)
        
        landmark = tf.cast(image_features['image/landmark'], tf.float32)
        landmark_batch.append(landmark)
 
    return image_batch,label_batch,bbox_batch,landmark_batch
