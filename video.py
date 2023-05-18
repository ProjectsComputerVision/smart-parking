import cv2
import numpy as np
import torch
import utils

#Read video
cap = cv2.VideoCapture("in_out.mp4")
out = cv2.VideoWriter('parking.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (1536, 864))

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # to RGB
    img = frame
    scale_percent = 80 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = utils.resize_image(img, width, height)
    mg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Inference
    results = utils.inference(img)
    # Draw results
    imgP = utils.imgProcess(img, results.xyxy[0].cpu().numpy())
    # to BGR
    imgP = cv2.cvtColor(imgP, cv2.COLOR_RGB2BGR)

    cv2.imshow('Frame', imgP)
    out.write(imgP)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
print("Done!")
cv2.destroyAllWindows()

