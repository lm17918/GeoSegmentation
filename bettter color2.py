import cv2
import numpy as np


def read_image():
    # Load the image in RGB
    image = cv2.imread("val/images/2012-04-26-Muenchen-Tunnel_4K0G0110.jpg")
    image = cv2.resize(image, (1080, 950))
    return image


def enhance_main_colors(image):
    # Reduce noise using Gaussian blur
    image_blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Convert the image from RGB to HSV
    hsv_image = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)

    # Define color ranges for main colors (red, green, blue, yellow, etc.)
    color_ranges = [
        # Red colors
        (np.array([0, 50, 50]), np.array([10, 255, 255])),  # Lower red
        (np.array([170, 50, 50]), np.array([180, 255, 255])),  # Upper red
        # Green colors
        (np.array([35, 50, 50]), np.array([85, 255, 255])),  # Green
        # Blue colors
        (np.array([100, 50, 50]), np.array([140, 255, 255])),  # Blue
        # Yellow colors
        (np.array([20, 50, 50]), np.array([35, 255, 255])),  # Yellow
        # Orange colors
        (np.array([10, 50, 50]), np.array([20, 255, 255])),  # Orange
        # Purple colors
        (np.array([140, 50, 50]), np.array([170, 255, 255])),  # Purple
    ]

    # Create an empty mask to combine all colors
    combined_mask = np.zeros_like(hsv_image[:, :, 0])

    # Iterate over all color ranges and create masks
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv_image, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)  # Combine the masks

    # Optional: Apply morphological operations to remove small variations and clean the mask
    kernel = np.ones((7, 7), np.uint8)  # Adjust the kernel size based on noise level
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, kernel
    )  # Closing to fill small holes
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, kernel
    )  # Opening to remove noise

    # Optional: Smooth the mask to reduce harsh edges
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # Apply the combined mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=combined_mask)

    return result_image, combined_mask


# Read and preprocess the image
image = read_image()

# Enhance the main colors in the image
result_image, combined_mask = enhance_main_colors(image)

# Show the resulting images
cv2.imwrite("OriginalImage.jpg", image)
cv2.imwrite("CombinedMask.jpg", combined_mask)
cv2.imwrite("ResultImage.jpg", result_image)
