import cv2
import numpy as np


def read_image():
    # Load the image in RGB
    image = cv2.imread("val/images/2012-04-26-Muenchen-Tunnel_4K0G0110.jpg")

    # image = cv2.resize(image, (1080, 950))
    return image


image = read_image()


# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for green colors
lower_green = np.array([30, 20, 20])  # Includes dull and dark greens
upper_green = np.array([90, 255, 255])

# Define the range for light brown colors
lower_brown = np.array([10, 50, 50])  # Light brown hues (orange to yellow tones)
upper_brown = np.array([25, 255, 200])

# Create masks for green and light brown
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)

# Combine the masks
combined_mask = cv2.bitwise_or(mask_green, mask_brown)

# Apply the combined mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

# Overlay the combined mask onto the original image
# Convert mask to 3 channels
mask_colored = cv2.merge([combined_mask, combined_mask, combined_mask])
overlay = cv2.addWeighted(image, 0.5, mask_colored, 0.5, 0)


# Save the output images (optional)
cv2.imwrite("green_brown_mask.jpg", combined_mask)
cv2.imwrite("overlay.jpg", overlay)
