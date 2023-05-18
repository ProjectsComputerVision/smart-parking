import yolov5
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
# Load model
model = yolov5.load('best.pt')

# Load image
img = cv2.imread('0006.png')
# to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Inference
results = model(img)

# Draw results
imgP = utils.imgProcess(img, results.xyxy[0].cpu().numpy())

# to BGR and imwrite
imgP = cv2.cvtColor(imgP, cv2.COLOR_RGB2BGR)
cv2.imwrite('0006_result.png', imgP)