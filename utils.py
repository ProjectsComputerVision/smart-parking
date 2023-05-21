import cv2
import numpy as np
import matplotlib.pyplot as plt
import yolov5

colors = {
    0: (46, 204, 112),
    1: (255, 121, 121)
}
labels = {0: 'car', 1: 'licence'}

def drawImg(img: np.ndarray, xmin, ymin, xmax, ymax, name, label):
    start_point = (int(xmin), int(ymin))
    end_point = (int(xmax), int(ymax))
    color = colors[label]

    img = cv2.rectangle(
        img,
        start_point, 
        end_point,
        color=color,
        thickness = 2
        )
    text_size, _ = cv2.getTextSize(
        name,
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.6,
        thickness=1
        )
    text_w, text_h = text_size

    img = cv2.rectangle(
        img,
        (int(xmin), int(ymin) - text_h - 5),
        (int(xmin) + text_w, int(ymin)),
        color=color,
        thickness=-1
        )
    img = cv2.putText(
        img,
        name,
        org=(int(xmin), int(ymin) - 5),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.6,
        color=(255, 255, 255),
        thickness=1,
        lineType=cv2.FILLED
        )
    return img

    
def imgProcess(img, data):

    for points in data:
        xmin, ymin, xmax, ymax, confidence,label = points
        if confidence > 0.5:
            name = labels[int(label)] + ' ' + str(round(confidence, 2))
            img = drawImg(img, xmin, ymin, xmax, ymax, name, int(label))
        
    return img

def inference(img):
    # Load model
    model = yolov5.load('best.pt')
    # inference
    results = model(img)
    return results

def resize_image(image, w,h):
    """
    Image resize with opencv
    """
    image = cv2.resize(image,(w, h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

