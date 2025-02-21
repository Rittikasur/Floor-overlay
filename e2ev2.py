import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

# -------------------------------
# ASSUMPTIONS:
# - scene_image: the original scene image (BGR) loaded from disk.
# - texture_image: the square texture image (BGR) to be tiled.
# - floor_mask: a binary mask (uint8, values 0 or 255) of the same size as scene_image
#   where the floor region is white (255) and the rest is black (0).

# For example, you might load them like:
# scene_image = cv2.imread("scene_image.jpg")
# texture_image = cv2.imread("texture_image.jpg")
# floor_mask = cv2.imread("floor_mask.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("D:/Quleep/Prototype/Code/mask_output/demo1.jpg", cv2.IMREAD_GRAYSCALE)
texture_image = cv2.imread("D:/Quleep/Prototype/Code/Data/floor4.jpg")
scene_image = cv2.imread("D:/Quleep/Prototype/Code/Data/image1.jpg")
# Apply binary threshold
_, floor_mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
# -------------------------------
# -------------------------------
# Ensure the mask is binary (0 and 1)
_, floor_mask_bin = cv2.threshold(floor_mask, 127, 255, cv2.THRESH_BINARY)
floor_mask_bin = (floor_mask_bin > 0).astype(np.uint8)

# Get coordinates (x, y) of floor pixels from the mask.
ys, xs = np.where(floor_mask_bin > 0)
if len(xs) < 3:
    raise ValueError("Not enough points in floor mask to compute a convex hull.")

# Create an array of coordinates (each row is [x, y])
floor_points = np.column_stack((xs, ys))

# Compute the convex hull of these points to get a good boundary for the floor.
hull_obj = ConvexHull(floor_points)
hull_indices = hull_obj.vertices
hull_points = floor_points[hull_indices]  # Destination points (in scene coordinates)

# Compute the bounding box of the hull to define the region of interest (ROI).
min_x = np.min(hull_points[:, 0])
max_x = np.max(hull_points[:, 0])
min_y = np.min(hull_points[:, 1])
max_y = np.max(hull_points[:, 1])

# Crop the floor ROI from the scene image.
floor_roi = scene_image[min_y:max_y, min_x:max_x]

# Create a tiled version of the texture image that covers the entire ROI.
roi_height, roi_width = floor_roi.shape[:2]
tex_h, tex_w = texture_image.shape[:2]
tiles_x = int(np.ceil(roi_width / tex_w))
tiles_y = int(np.ceil(roi_height / tex_h))
tiled_texture = np.tile(texture_image, (tiles_y, tiles_x, 1))
tiled_texture = tiled_texture[:roi_height, :roi_width]

# For mapping, compute the source coordinates in the texture image.
# We subtract the ROI top-left (min_x, min_y) from the hull points.
source_hull_points = hull_points.astype(np.float32).copy()
source_hull_points[:, 0] -= min_x
source_hull_points[:, 1] -= min_y

# Use Delaunay triangulation on the hull points (destination) to subdivide the floor.
tri = Delaunay(hull_points)

# Prepare an image for the warped floor region.
warped_floor = np.zeros_like(floor_roi)

def warp_triangle(src_img, src_tri, dst_tri, dst_img):
    """
    Warps a triangular region from src_img defined by src_tri into dst_img at dst_tri.
    """
    # Compute bounding rectangle for destination triangle.
    r_dst = cv2.boundingRect(np.float32([dst_tri]))
    # Offset destination triangle points relative to the bounding rectangle.
    dst_tri_offset = np.array([[pt[0] - r_dst[0], pt[1] - r_dst[1]] for pt in dst_tri], dtype=np.float32)
    
    # Compute bounding rectangle for source triangle.
    r_src = cv2.boundingRect(np.float32([src_tri]))
    src_tri_offset = np.array([[pt[0] - r_src[0], pt[1] - r_src[1]] for pt in src_tri], dtype=np.float32)
    
    # Crop the source patch.
    src_patch = src_img[r_src[1]:r_src[1]+r_src[3], r_src[0]:r_src[0]+r_src[2]]
    
    # Compute the affine transform.
    warp_mat = cv2.getAffineTransform(src_tri_offset, dst_tri_offset)
    dst_patch = cv2.warpAffine(src_patch, warp_mat, (r_dst[2], r_dst[3]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    # Create a mask for the destination triangle.
    mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_offset), (1, 1, 1), 16, 0)
    
    # Get the destination image section.
    dst_img_section = dst_img[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]]
    
    # If there is any mismatch in sizes, adjust by taking the minimal common region.
    h_dst, w_dst = dst_img_section.shape[:2]
    h_patch, w_patch = dst_patch.shape[:2]
    h_min = min(h_dst, h_patch)
    w_min = min(w_dst, w_patch)
    
    dst_img_section = dst_img_section[:h_min, :w_min]
    dst_patch = dst_patch[:h_min, :w_min]
    mask = mask[:h_min, :w_min]
    
    # Blend the warped patch with the destination patch.
    # Convert to float for accurate blending and convert back to uint8.
    blended = (dst_img_section.astype(np.float32) * (1 - mask.astype(np.float32)) +
               dst_patch.astype(np.float32) * mask.astype(np.float32))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    dst_img[r_dst[1]:r_dst[1]+h_min, r_dst[0]:r_dst[0]+w_min] = blended

# Warp each triangle from the tiled texture into the corresponding triangle in the floor ROI.
for simplex in tri.simplices:
    # Destination triangle (in scene coordinates)
    dst_tri = hull_points[simplex].astype(np.float32)
    # Convert destination triangle coordinates to ROI coordinates.
    dst_tri_roi = dst_tri.copy()
    dst_tri_roi[:, 0] -= min_x
    dst_tri_roi[:, 1] -= min_y

    # Source triangle from the tiled texture (precomputed source hull points)
    src_tri = source_hull_points[simplex].astype(np.float32)
    
    warp_triangle(tiled_texture, src_tri, dst_tri_roi, warped_floor)

# Insert the warped floor texture back into the original scene image.
final_output = scene_image.copy()
final_output[min_y:max_y, min_x:max_x] = warped_floor

# Optionally, blend the warped floor with the original scene for a more natural composite.
blended_result = cv2.addWeighted(scene_image, 0.5, final_output, 0.5, 0)

# Display the results.
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB))
plt.title("Original Scene Image")

plt.subplot(1, 3, 2)
plt.imshow(floor_mask_bin, cmap='gray')
plt.title("Floor Mask")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(blended_result, cv2.COLOR_BGR2RGB))
plt.title("Final Warped Texture on Floor")
plt.tight_layout()
plt.show()
