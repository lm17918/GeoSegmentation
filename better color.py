import cv2
import numpy as np
from utils import read_image, overlay_mask_onto_original_image
from color import create_trees_mask, create_street_mask

image = read_image()

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Combine the masks
tree_mask = create_trees_mask(hsv_image)

street_mask = create_street_mask(hsv_image)


# Apply the combined mask to the original image
# masked_image = cv2.bitwise_and(image, image, mask=combined_mask)


overlay_tree = overlay_mask_onto_original_image(image, tree_mask)
overlay_street = overlay_mask_onto_original_image(image, street_mask)

# Save the output images (optional)
cv2.imwrite("street_mask.jpg", street_mask)
cv2.imwrite("overlay_tree.jpg", overlay_tree)
cv2.imwrite("overlay_street.jpg", overlay_street)
# cv2.imwrite("masked_image.jpg", masked_image)
