import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Load the square image
image = cv2.imread("D:/Quleep/Prototype/Code/Data/floor4.jpg")  # Replace with actual image path
h, w = image.shape[:2]

# Step 1: Define source (square image) and destination (irregular shape) points
src_pts = np.array([
    [0, 0],       # Top-left
    [w-1, 0],     # Top-right
    [w-1, h-1],   # Bottom-right
    [0, h-1]      # Bottom-left
])

# Four main boundary points for homography
dst_pts = np.array([
    [0, 111],  # Top-left
    [297, 128],  # Top-right
    [297, 168],  # Bottom-right
    [0, 168]     # Bottom-left
])

# Compute Homography
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Warp the image using perspective transformation
output_width = int(max(dst_pts[:, 0]))
output_height = int(max(dst_pts[:, 1]))
warped_image = cv2.warpPerspective(image, H, (output_width, output_height))

# Step 2: Apply Delaunay triangulation on the warped space
extra_dst_pts = np.array([
    [240, 106], [255, 109], [280, 117], [297, 128], [0, 111],
    [1, 110], [4, 109], [16, 106], [58, 105]
], dtype=np.float32)

# Combine main boundary + extra points for triangulation
all_dst_pts = np.vstack((dst_pts, extra_dst_pts))

# Perform Delaunay triangulation
tri = Delaunay(all_dst_pts)

# Create an empty image to hold the final warped output
final_output = np.zeros((output_height, output_width, 3), dtype=np.uint8)

# Function to warp each triangle affinely
def warp_triangle(img, src_tri, dst_tri, output):
    matrix = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    warped = cv2.warpAffine(img, matrix, (output.shape[1], output.shape[0]))
    
    # Create a mask for the triangle
    mask = np.zeros_like(output, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_tri), (255, 255, 255))
    
    # Combine warped triangle into final output
    output[np.where(mask != 0)] = warped[np.where(mask != 0)]

# Loop through each triangle and apply affine transformation
for simplex in tri.simplices:
    src_tri = src_pts[simplex % 4]  # Source: Original quadrilateral
    dst_tri = all_dst_pts[simplex]  # Destination: Triangulated irregular shape

    warp_triangle(warped_image, src_tri, dst_tri, final_output)

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.title("Warped Image (Perspective)")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
plt.title("Final Warped Image (Perspective + Mesh)")

plt.show()
