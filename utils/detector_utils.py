# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict


detection_graph = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.27

MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_toad_on_image(box, image_np, toad):
    im_width = image_np.shape[1]
    im_height = image_np.shape[0]
    (left, right, top, bottom) = (box[1] * im_width, box[3] * im_width, box[0] * im_height, box[2] * im_height)
    perc0 = max(image_np.shape)/max(toad.shape)
    w0 = int(toad.shape[1]*perc0)
    h0 = int(toad.shape[0]*perc0)
    toad = cv2.resize(toad, (h0, w0))
    perc = 0.4
    w = int(toad.shape[1]*perc)
    h = int(toad.shape[0]*perc)
    toad = cv2.resize(toad, (h, w))
    x = int((left+right)/2 - toad.shape[1]/2)
    y = int((top+bottom)/2)
    best_image = overlay_image(image_np, toad, (x, y))
    return best_image

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)

def overlay_image(bg, fg, coords):
    # определить размер переднего плана (ширина, высота) и
    # координаты его размещения
    (sH, sW) = fg.shape[:2]
    (x, y) = coords

    # наложение должно быть точно такой ширины и высоты как
    # исходная картинка, но полностью пустым, *кроме* переднего
    # плана, который мы добавляем
    overlay = np.zeros(bg.shape, dtype="uint8")
    overlay[y:y + sH, x:x + sW] = fg

    # альфа-канал контролирует, *координаты* и *степень*
    # прозрачности, его размеры такие же, как у исходного
    # изображения, но он содержит только маску наложения
    alpha = np.zeros(bg.shape[:2], dtype="uint8")
    dwiMask = cv2.cvtColor(fg, cv2.COLOR_BGRA2GRAY)
    dwiMask = cv2.threshold(dwiMask, 0, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((6, 6), np.uint8)
    op = cv2.erode(dwiMask, kernel, iterations = 1)
    kernel1 = np.ones((7, 7), np.uint8)
    op = cv2.morphologyEx(op, cv2.MORPH_OPEN, kernel1)
    alpha[y:y + sH, x:x + sW] = op
    alpha = np.dstack([alpha] * 3)

    # выполняем альфа-смешивание для переднего плана,
    # фона и альфа-канала
    output = alpha_blend(overlay, bg, alpha)

    # возвращаем результат
    return output


def alpha_blend(fg, bg, alpha):
    # преобразуем фон, передний план и альфа-канал
    # в числа с плавающей запятой в диапазоне [0, 1]
    fg = fg.astype("float")
    bg = bg.astype("float")
    alpha = alpha.astype("float") / 255

    # выполняем альфа-смешивание
    fg = cv2.multiply(fg, alpha)
    bg = cv2.multiply(bg, 1-alpha)

    # добавляем передний план и фон, получая конечный результат
    output = cv2.add(fg, bg)

    # возвращаем результат
    return output.astype("uint8")

def draw_box_on_image(num_hands_detect, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
    return image_np

def prepare_image(image):
    n_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]-1))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j, 3] > 0:
                for k in range(image.shape[2] - 1):
                    n_image[i, j, k] = image[i, j, k]
            else:
                for k in range(image.shape[2] - 1):
                    n_image[i, j, k] = 0
    return n_image.astype(np.float32)
