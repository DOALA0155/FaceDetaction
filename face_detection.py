import numpy as np
import cv2
import matplotlib.pyplot as plt
from functions import *

image_name = "face1.jpg"
image = cv2.imread("./original_images/face1.jpg".format(image_name))
gray_image = convert_to_gray(image)

faces = detection_face(gray_image)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, x + h), (255, 0, 0), 2)

cv2.imwrite("./result_images/{}".format(image_name), image)
