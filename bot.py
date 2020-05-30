from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import os
import numpy as np
import urllib.request

import telebot

score_thresh = 0.2
num_hands_detect = 1
result_storage_path = 'tmp'

def save_image(message):
    cid = message.chat.id
    image_id = get_image_id_from_message(message)
    bot.send_message(cid, 'Подбираем легушьку для вас...')
    file_path = bot.get_file(image_id).file_path
    image_url = "https://api.telegram.org/file/bot{0}/{1}".format(token, file_path)
    if not os.path.exists(result_storage_path):
        os.makedirs(result_storage_path)
    image_name = "{0}.jpg".format(image_id)
    urllib.request.urlretrieve(image_url, "{0}/{1}".format(result_storage_path, image_name))

    return image_name

def get_image_id_from_message(message):
    return message.photo[len(message.photo)-1].file_id

def image_processing(image_name):
    image_np = cv2.imread('./tmp/'+image_name)
    toad = cv2.imread('./data/'+np.random.choice(os.listdir('./data/')), cv2.IMREAD_UNCHANGED)
    boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
    best_box = boxes[np.argmax(scores)]
    new_image = detector_utils.draw_box_on_image(num_hands_detect, boxes, image_np)
    best_image = detector_utils.draw_toad_on_image(best_box, image_np, toad)
    cv2.imwrite(image_name, best_image)
    return best_image



detection_graph, sess = detector_utils.load_inference_graph()

num_hands_detect = 1


bot = telebot.TeleBot(token)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/help":
        bot.send_message(message.from_user.id, "Отправь мне фотографию ладошки!")

@bot.message_handler(content_types=['photo'])
def get_photo_message(message):
    img_name = save_image(message)
    img = image_processing(img_name)
    bot.send_photo(message.chat.id, open(img_name, 'rb'))

bot.polling(none_stop=True, interval=0)

