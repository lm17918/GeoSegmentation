import cv2
import numpy as np


def create_trees_mask(image):
    # Define the range for green colors
    lower_green = np.array([30, 20, 20])  # Includes dull and dark greens
    upper_green = np.array([90, 255, 255])

    # Define the range for light brown colors
    lower_brown = np.array([10, 50, 50])  # Light brown hues (orange to yellow tones)
    upper_brown = np.array([25, 255, 200])

    # Create masks for green and light brown
    mask_green = cv2.inRange(image, lower_green, upper_green)
    mask_brown = cv2.inRange(image, lower_brown, upper_brown)

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_green, mask_brown)
    return combined_mask


def create_street_mask(image):
    # Define the range for dark gray to black (asphalt-like roads)
    lower_gray = np.array([0, 0, 30])  # Dark gray to black
    upper_gray = np.array([180, 50, 120])

    # Define the range for light gray to white (concrete roads, sidewalks)
    lower_light_gray = np.array([0, 0, 120])
    upper_light_gray = np.array([180, 50, 255])

    # Define the range for beige to light brown (some roads may have these tones)
    lower_beige = np.array([10, 20, 150])
    upper_beige = np.array([30, 100, 255])

    # Create masks for each road-related color
    mask_gray = cv2.inRange(image, lower_gray, upper_gray)
    mask_light_gray = cv2.inRange(image, lower_light_gray, upper_light_gray)
    mask_beige = cv2.inRange(image, lower_beige, upper_beige)

    # Combine all road masks
    combined_road_mask = cv2.bitwise_or(mask_gray, mask_light_gray)
    combined_road_mask = cv2.bitwise_or(combined_road_mask, mask_beige)
    return combined_road_mask
