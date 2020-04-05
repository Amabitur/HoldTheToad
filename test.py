import cv2
import numpy as np
import os
import imutils

pth = '/home/deer/Pictures/toads'
pth1 = '/home/deer/Pictures/toads/new'
path = list(os.walk(pth))

for file in path[0][2]:
    image = cv2.imread(pth+'/'+file)
    size = max(image.shape)
    n_image = np.full((size, size, 3), 255).astype(np.float32)
    n_image[0:image.shape[0], 0:image.shape[1]] = image
    n_image = cv2.resize(n_image, (1024, 1024))
    cv2.imwrite(pth1+'/'+file.replace('.jpg', '-n.jpg'), n_image)
    #n_image_15 = imutils.rotate_bound(n_image, 15)
    #cv2.imwrite(pth1 + '/' + file.replace('.jpg', '-n15.jpg'), n_image_15)
    #n_image__15 = imutils.rotate_bound(n_image, -15)
    #cv2.imwrite(pth1 + '/' + file.replace('.jpg', '-n_15.jpg'), n_image__15)

    nn_image = cv2.flip(n_image, 1)
    cv2.imwrite(pth1 + '/' + file.replace('.jpg', '-nn.jpg'), nn_image)
