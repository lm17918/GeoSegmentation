import cv2
import numpy as np


def read_image():
    # Load the image in RGB
    image = cv2.imread("data/val/images/2012-04-26-Muenchen-Tunnel_4K0G0110.jpg")

    # image = cv2.resize(image, (1080, 950))
    return image


def overlay_mask_onto_original_image(image, combined_mask):
    # Convert mask to 3 channels
    mask_colored = cv2.merge([combined_mask, combined_mask, combined_mask])
    overlay = cv2.addWeighted(image, 0.5, mask_colored, 0.5, 0)
    return overlay
