import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import tensorflow as tf
import numpy as np
from PIL import Image

def compute_gradcam(image, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_colored_resize = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored_resize, alpha, 0)
    return overlay

def visualize_gradcam(image_path, model, last_conv_layer_name, img_height=224, img_width=224):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    heatmap = compute_gradcam(img_tensor, model, last_conv_layer_name)
    img_array_uint8 = (img_array * 255).astype(np.uint8)
    overlay = overlay_heatmap(heatmap, img_array_uint8)

    return overlay
