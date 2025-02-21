import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Load the square image
image = cv2.imread("D:/Quleep/Prototype/Code/Data/floor4.jpg")  # Replace with actual image path
h, w = image.shape[:2]

# Define the destination points (irregular shape boundary)
dst_pts = np.array([
    [219, 105], [240, 106], [255, 109], [280, 117], [296, 127],
    [297, 168], [0, 168], [0, 111], [1, 110], [4, 109],
    [16, 106], [58, 105]
])

# Generate source points as a grid inside the square image
# We map them proportionally to the width and height of the original image
num_pts = len(dst_pts)
src_pts = np.array([
    [w * (i / (num_pts - 1)), h * (i / (num_pts - 1))]
    for i in range(num_pts)
])

# Compute Delaunay triangulation on the destination shape
tri = Delaunay(dst_pts)

# Create an empty output image
output_size = (max(dst_pts[:, 0]), max(dst_pts[:, 1]))
warped_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

# Function to warp each triangle
def warp_triangle(img, src_tri, dst_tri, output):
    # Compute affine transform
    matrix = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    
    # Warp the small triangle
    warped = cv2.warpAffine(img, matrix, (output.shape[1], output.shape[0]))
    
    # Create a mask for the triangle
    mask = np.zeros_like(output, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_tri), (255, 255, 255))
    
    # Combine warped triangle with final image
    output[np.where(mask != 0)] = warped[np.where(mask != 0)]

# Loop through each triangle and apply the affine warp
for simplex in tri.simplices:
    src_tri = src_pts[simplex]  # Use corresponding source points
    dst_tri = dst_pts[simplex]
    
    warp_triangle(image, src_tri, dst_tri, warped_image)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.title("Warped Image (Triangular Mesh)")

plt.show()
