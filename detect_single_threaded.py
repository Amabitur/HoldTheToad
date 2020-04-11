from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import os
import numpy as np

detection_graph, sess = detector_utils.load_inference_graph()

score_thresh=0.2
num_hands_detect = 1

image_np = cv2.imread('/home/deer/Pictures/photo_2020-04-05_18-36-28.jpg')
toad = cv2.imread('/home/deer/Pictures/toads/new/44695610-european-toad-b-bufo-isolated-on-white-n.jpg')
#toad = detector_utils.prepare_image(toad)
print(toad.shape)
cv2.imwrite('toad.png', toad)

boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)

best_box = boxes[np.argmax(scores)]

new_image = detector_utils.draw_box_on_image(num_hands_detect, boxes, image_np)
best_image = detector_utils.draw_toad_on_image(best_box, new_image, toad)
cv2.imwrite('haha.png', best_image)
