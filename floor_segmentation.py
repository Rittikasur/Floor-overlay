import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
image = cv2.imread('D:/Quleep/Prototype/Code/Data/image2.jpg')  # Replace with the path to your image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection to find edges
edges = cv2.Canny(blurred, threshold1=40, threshold2=140) #original threshold 50, 150

# Apply morphological operations to close gaps in the edges
kernel = np.ones((5, 5), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours from the processed edges
contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask to highlight the floor
mask = np.zeros_like(gray)

# Filter and draw only the largest contour, assuming it represents the floor
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

# Apply the mask to the original image
segmented_floor = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Segmented Floor')
plt.imshow(segmented_floor)
plt.axis('off')

plt.tight_layout()
plt.show()
