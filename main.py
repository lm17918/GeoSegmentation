import cv2
import numpy as np


def save_img(output_path, cleaned):
    # Save preprocessed image
    cv2.imwrite(output_path, cleaned)


def preprocess_image(input_path: str, output_path: str) -> None:
    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not find the image: {input_path}")
    image = cv2.resize(image, (1080, 950))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    save_img("preprocessed_image.jpg", gray)

    # Histogram equalization
    equalized = cv2.equalizeHist(gray)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

    save_img("preprocessed_image2.jpg", blurred)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    save_img("preprocessed_image3.jpg", edges)

    # Reduce kernel size for morphological operations
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 3)
    )  # Smaller kernel to reduce thickness
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    save_img("preprocessed_image4.jpg", cleaned)

    # Reduce number of dilation iterations to prevent thickening
    dilated = cv2.dilate(
        edges, kernel, iterations=1
    )  # Reduce iterations for thinner lines
    save_img("preprocessed_image5.jpg", dilated)

    # Optionally, apply a Gaussian blur to smooth the dilated image
    smoothed = cv2.GaussianBlur(dilated, (5, 5), 0)
    save_img("preprocessed_image5_smoothed.jpg", smoothed)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)

    # Create an empty mask for large objects
    mask = np.zeros_like(gray, dtype=np.uint8)

    # Minimum size and aspect ratio thresholds
    min_size = 1000  # Minimum area (in pixels)
    min_aspect_ratio = 0.5  # Minimum aspect ratio (height/width or width/height)

    # Iterate over each component
    for i in range(1, num_labels):  # Skip the background (label 0)
        x, y, w, h, area = stats[i]
        if area >= min_size:
            # Calculate aspect ratio (height/width)
            aspect_ratio = min(w, h) / max(w, h)

            if aspect_ratio >= min_aspect_ratio:
                mask[labels == i] = 255  # Keep the component with valid aspect ratio

    # Apply the mask to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    save_img("preprocessed_image6.jpg", filtered_image)


# Input and output image paths
input_image_path = "val/images/2012-04-26-Muenchen-Tunnel_4K0G0110.jpg"  # Change to the path of your input image
output_image_path = "preprocessed_image.jpg"  # Change to your desired output path

# Preprocess the image
preprocess_image(input_image_path, output_image_path)
