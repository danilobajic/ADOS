import cv2
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, IntSlider
import os

# Define the path to the image
image_path = 'dataset/test/images/00000059_jpg.rf.755e9674fa8550971918a53b8edffa70.jpg'

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: The image file at {image_path} does not exist.")
else:
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read the image file at {image_path}.")
    else:
        def adjust_hsv(hue, hue_range, saturation, saturation_range, value, value_range):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower_bound = np.array([hue - hue_range, saturation - saturation_range, value - value_range])
            upper_bound = np.array([hue + hue_range, saturation + saturation_range, value + value_range])

            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            result = cv2.bitwise_and(img, img, mask=mask)

            plt.figure(figsize=(12, 6))
            plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
            plt.axis('off')
            plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Detected Objects')
            plt.axis('off')
            plt.show()

            return mask

        def draw_bounding_boxes(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            img_with_boxes = img.copy()

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            plt.figure(figsize=(18, 6))
            plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
            plt.axis('off')
            plt.subplot(132), plt.imshow(mask, cmap='gray'), plt.title('Mask')
            plt.axis('off')
            plt.subplot(133), plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)), plt.title('Bounding Boxes')
            plt.axis('off')
            plt.show()

        def interactive_adjustment(hue, hue_range, saturation, saturation_range, value, value_range):
            mask = adjust_hsv(hue, hue_range, saturation, saturation_range, value, value_range)
            draw_bounding_boxes(mask)

        interact(interactive_adjustment,
                 hue=IntSlider(min=0, max=180, step=1, value=20, description='Hue'),
                 hue_range=IntSlider(min=0, max=50, step=1, value=12, description='Hue Range'),
                 saturation=IntSlider(min=0, max=255, step=1, value=240, description='Saturation'),
                 saturation_range=IntSlider(min=0, max=100, step=1, value=100, description='Saturation Range'),
                 value=IntSlider(min=0, max=255, step=1, value=170, description='Value'),
                 value_range=IntSlider(min=0, max=100, step=1, value=100, description='Value Range'))
