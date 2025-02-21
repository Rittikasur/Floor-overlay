import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="int32")
    points = np.array(corners)
    # Sort points based on Y-values
    sorted_points = points[np.argsort(points[:, 1])]

    # Extract sorted Y-coordinates
    y_coords = sorted_points[:, 1]

    # Compute differences between consecutive Y-values
    y_diffs = np.diff(y_coords)

    # Set a dynamic threshold as the maximum gap in Y-values
    threshold_index = np.argmax(y_diffs)  # Find the largest jump in Y-values
    split_value = y_coords[threshold_index]  # The Y-value where the split occurs

    # Split the points into top and bottom groups
    top_points = sorted_points[sorted_points[:, 1] <= split_value]
    bottom_points = sorted_points[sorted_points[:, 1] > split_value]

    # Print results
    print("Top Points:\n", top_points)
    print("Bottom Points:\n", bottom_points)

    # Sort top points by Y (ascending)
    top_sorted = top_points[np.argsort(top_points[:, 1])]
    
    # Sort bottom points by Y (descending)
    bottom_sorted = bottom_points[np.argsort(bottom_points[:, 1])[::-1]]
    
    # Find corners
    top_left = min(top_sorted, key=lambda p: p[0])  # Smallest X in top [ 58, 105] # 
    top_right = max(top_sorted, key=lambda p: p[0])  # Largest X in top [219 ,105] # 
    bottom_left = min(bottom_sorted, key=lambda p: p[0])  # Smallest X in bottom
    bottom_right = max(bottom_sorted, key=lambda p: p[0])  # Largest X in bottom

    # # Find leftmost and rightmost points for both groups
    # top_left = top_points[np.argmin(top_points[:, 0])]
    # top_right = top_points[np.argmax(top_points[:, 0])]
    # bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
    # bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]

    # Print results
    print("Top Left:", top_left)
    print("Top Right:", top_right)
    print("Bottom Left:", bottom_left)
    print("Bottom Right:", bottom_right)

    rect[0] = top_left  # Top-left
    rect[2] = bottom_right  # Bottom-right

    rect[1] = top_right  # Top-right
    rect[3] = bottom_left  # Bottom-left

    return rect


# Read the segmentation mask (ensure it's binary)
mask = cv2.imread("D:/Quleep/Prototype/Code/mask_output/demo1.jpg", cv2.IMREAD_GRAYSCALE)
carpet = cv2.imread("D:/Quleep/Prototype/Code/Data/floor4.jpg")
original_image = cv2.imread("D:/Quleep/Prototype/Code/Data/image1.jpg")
# Apply binary threshold
_, binary_mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
approx = None
# Check if any contour is found
if contours:
    # Get the largest contour (assuming it's the floor)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate a quadrilateral
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.convexHull(largest_contour) # cv2.approxPolyDP(largest_contour, epsilon, True)

    # Convert to RGB to display colors
    result = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Draw the approximated polygon
    cv2.polylines(result, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw yellow circles on the corners
    for point in approx:
        x, y = point.ravel()
        cv2.circle(result, (x, y), 5, (0, 255, 255), -1)  # Yellow circle

    # Show the image
    cv2.imshow("Corners Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found!")

corners = approx.reshape(-1, 2)
print(corners)

ordered_corners = order_points(corners)
print(ordered_corners)
result = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
for point in ordered_corners:
    x, y = point.ravel()
    print((int(x), int(y)))
    cv2.circle(result, (int(x), int(y)), 5, (0, 255, 255), -1)  # Yellow circle
cv2.imshow("Corners Detection 2", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define source points (carpet corners in the original image)
h, w = carpet.shape[:2]
src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

# Compute the homography matrix
H, _ = cv2.findHomography(src_pts, ordered_corners)

# warped_carpet = cv2.warpPerspective(carpet, H, (mask.shape[1], mask.shape[0]))
# result = original_image.copy()
# mask_inv = cv2.bitwise_not(binary_mask)
# result = cv2.bitwise_and(result, result, mask=mask_inv)  # Remove original floor
# result = cv2.add(result, warped_carpet)  # Add transformed carpet

# cv2.imshow("Corners Detection 2", warped_carpet)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#-------------------------------------------------------------------------------------------------------------
# Load images
# mask = cv2.imread("mask.jpg")  # Floor segmentation mask
# tile = cv2.imread("floor2.jpg")  # Tile texture to overlay

# # Define source points (covering an entire tile region)
# tile_h, tile_w = tile.shape[:2]
# src_pts = np.array([[0, 0], [tile_w, 0], [tile_w, tile_h], [0, tile_h]], dtype=np.float32)

# # Compute the Homography matrix
# H, _ = cv2.findHomography(src_pts, ordered_corners)

# # Create a tiled image large enough to cover the floor
# tile_repeat_x = 8  # Adjust based on required tiling
# tile_repeat_y = 8
# tiled_image = np.tile(tile, (tile_repeat_y, tile_repeat_x, 1))

# # Warp the tiled image to fit the floor
# warped_tile = cv2.warpPerspective(tiled_image, H, (mask.shape[1], mask.shape[0]))

# # Create a binary mask for overlay
# binary_mask = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)[1]
# binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel

# # Apply the warped tile only where the floor is segmented
# result = np.where(binary_mask == 255, warped_tile, mask)

# # Show result
# cv2.imshow("Warped Tile on Floor", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#------------------------------------------------------------------------------------------------------------



#Load images
tile = cv2.imread("D:/Quleep/Prototype/Code/Data/floor4.jpg")  # Tile texture to overlay
mask = cv2.imread("D:/Quleep/Prototype/Code/mask_output/demo1.jpg")
binary_mask = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)[1]
# Define source points from the tile image
# Create a larger tiled image (to ensure full coverage)
tile_repeat_x = 2  # Increase number of repetitions
tile_repeat_y = 2
tiled_image = np.tile(tile, (tile_repeat_y, tile_repeat_x, 1))
tile_h, tile_w = tiled_image.shape[:2]
src_pts = np.array([[0, 0], [tile_w, 0], [tile_w, tile_h], [0, tile_h]], dtype=np.float32)

# Compute Homography
H, _ = cv2.findHomography(src_pts, ordered_corners)

# Warp the tiled image onto the floor
warped_tile = cv2.warpPerspective(tiled_image, H, (mask.shape[1], mask.shape[0]))
carpet_msak = cv2.bitwise_not(cv2.cvtColor(warped_tile, cv2.COLOR_BGR2GRAY))
carpet_mask_binary = cv2.threshold(carpet_msak,  250, 255, cv2.THRESH_BINARY)[1]
# Find uncovered areas where homography didn't reach
uncovered_mask = cv2.bitwise_and(binary_mask, carpet_mask_binary)

# Extend warped tiles into uncovered regions (Doesn't Work)
# while cv2.countNonZero(uncovered_mask) > 0:
#     extra_warped = cv2.warpPerspective(tiled_image, H, (mask.shape[1], mask.shape[0]))
#     warped_tile = np.where(uncovered_mask[:, :, None] > 0, extra_warped, warped_tile)
#     carpet_msak = cv2.bitwise_not(cv2.cvtColor(warped_tile, cv2.COLOR_BGR2GRAY))
#     carpet_mask_binary = cv2.threshold(carpet_msak,  250, 255, cv2.THRESH_BINARY)[1]
#     uncovered_mask = cv2.bitwise_and(binary_mask,carpet_mask_binary )

# Overlay the final warped tile image onto the floor mask
binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
result = np.where(binary_mask == 255, warped_tile, mask)

uncovered_mask = cv2.cvtColor(uncovered_mask, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
resized_mask = cv2.resize(tiled_image, (uncovered_mask.shape[1], uncovered_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
tresult = np.where(uncovered_mask == 255, resized_mask, warped_tile)
roomimg = np.where(binary_mask == 255, tresult, original_image)
# Show result
# cv2.imwrite('D:/Quleep/Prototype/Code/Data/output/final2_2.jpg', roomimg)
cv2.imshow("Final tresult", tresult)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Final uncovered_mask", uncovered_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Final result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Final binary_mask", binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Final resized_mask", resized_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Final warped_tile", warped_tile)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("Final Warped Tile", roomimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(ordered_corners)
