import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- ASSUMPTIONS: ---
# scene_image: original scene image (BGR)
# texture_image: single tile texture image (BGR)
# floor_mask: binary mask (0/255) with the floor region white

# Example loading (uncomment and adjust paths as needed):
# scene_image = cv2.imread("scene_image.jpg")
# texture_image = cv2.imread("texture_image.jpg")
# floor_mask = cv2.imread("floor_mask.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("D:/Quleep/Prototype/Code/mask_output/demo1.jpg", cv2.IMREAD_GRAYSCALE)
texture_image = cv2.imread("D:/Quleep/Prototype/Code/Data/floor4.jpg")
scene_image = cv2.imread("D:/Quleep/Prototype/Code/Data/image1.jpg")
# Apply binary threshold
_, floor_mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
# --- Ensure the mask is binary ---
_, floor_mask_bin = cv2.threshold(floor_mask, 127, 255, cv2.THRESH_BINARY)

# --- Extract the floor region as a quadrilateral ---
# Get (x, y) coordinates of white pixels in the mask.
ys, xs = np.where(floor_mask_bin > 0)
if len(xs) < 3:
    raise ValueError("Not enough points in floor mask to compute a quadrilateral.")

# Stack coordinates (x, y)
floor_points = np.column_stack((xs, ys)).astype(np.float32)

# Compute the convex hull.
hull = cv2.convexHull(floor_points)

# Approximate the hull with a polygon.
epsilon = 0.02 * cv2.arcLength(hull, True)
approx = cv2.approxPolyDP(hull, epsilon, True)

# If not a quadrilateral, fall back to a minimum-area rectangle.
if len(approx) != 4:
    rect = cv2.minAreaRect(floor_points)
    approx = cv2.boxPoints(rect)
    approx = np.int0(approx)
else:
    approx = approx.reshape(4, 2)

dst_pts = approx.astype(np.float32)

# --- Order the Points (top-left, top-right, bottom-right, bottom-left) ---
def order_points(pts):
    # Sort by x-coordinate
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # Order left-most by y to get top-left and bottom-left.
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    tl, bl = leftMost
    # Order right-most by y to get top-right and bottom-right.
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    tr, br = rightMost
    return np.array([tl, tr, br, bl], dtype=np.float32)

dst_pts = order_points(dst_pts)

# --- Create a Tiled Pattern from the Texture ---
# Define how many times you want the tile repeated.
tile_h, tile_w = texture_image.shape[:2]
num_tiles_x = 4  # Number of tiles horizontally
num_tiles_y = 4  # Number of tiles vertically

# Create the pattern by tiling the texture image.
pattern = np.tile(texture_image, (num_tiles_y, num_tiles_x, 1))
pattern_h, pattern_w = pattern.shape[:2]

# --- Define Source Points for the Pattern ---
src_pts = np.array([
    [0, 0],
    [pattern_w - 1, 0],
    [pattern_w - 1, pattern_h - 1],
    [0, pattern_h - 1]
], dtype=np.float32)

# --- Compute the Homography ---
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Warp the pattern image to the size of the scene image.
warped_pattern = cv2.warpPerspective(pattern, H, (scene_image.shape[1], scene_image.shape[0]))

# --- Blend the Warped Pattern with the Scene ---
# Create a 3-channel version of the floor mask.
floor_mask_3ch = cv2.merge([floor_mask_bin, floor_mask_bin, floor_mask_bin])
# Replace only the floor area in the scene with the warped tiled pattern.
final_output = np.where(floor_mask_3ch == 255, warped_pattern, scene_image)

# --- Display the Result ---
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
plt.title("Scene with Tiled, Perspective-Warped Texture on Floor")
plt.axis('off')
plt.show()
