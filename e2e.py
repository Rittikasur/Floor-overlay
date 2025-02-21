import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Normalize
from scipy.spatial import ConvexHull, Delaunay
from segment_anything import SamPredictor, sam_model_registry

target_width, target_height = 640, 480

# Load the square image to be tiled
texture_image = cv2.imread("D:/Quleep/Prototype/Code/Data/floor4.jpg")
h, w = texture_image.shape[:2]

# Load the target scene image (with a floor)
scene_image = cv2.imread("D:/Quleep/Prototype/Code/Data/image1.jpg")
scene_image = cv2.resize(scene_image, (target_width, target_height))

# Step 1: Load MiDaS for depth estimation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Estimate depth map
scene_rgb = cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB)
input_tensor = transform(scene_rgb).unsqueeze(0).to(device)
with torch.no_grad():
    depth_map = midas(input_tensor).squeeze().cpu().numpy()

# Resize and normalize depth map to the scene image size
depth_map = cv2.resize(depth_map, (scene_image.shape[1], scene_image.shape[0]))
depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))


# Step 2: Segment the floor using SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth").to(device)
predictor = SamPredictor(sam)
predictor.set_image(scene_image)

# User provides points to mark the floor area
floor_mask = predictor.predict(point_coords=np.array([[400, 500]]), point_labels=np.array([1]))[0]

# Step 3: Extract 3D Correspondences for the Floor
y_idxs, x_idxs = np.where(floor_mask > 0)
z_values = depth_map[y_idxs, x_idxs]
floor_pts_3D = np.stack((x_idxs, y_idxs, z_values), axis=1)

# Compute Convex Hull for a more accurate floor boundary
hull = ConvexHull(floor_pts_3D[:, :2])
hull_points = floor_pts_3D[hull.vertices, :2]

# Step 4: Use Delaunay Triangulation for Irregular Shape Handling
tri = Delaunay(hull_points)

# Create an empty image for the final result
final_output = np.zeros_like(scene_image)

# Function to warp triangles
def warp_triangle(img, src_tri, dst_tri, output):
    matrix = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    warped = cv2.warpAffine(img, matrix, (output.shape[1], output.shape[0]))
    
    # Create mask for the triangle
    mask = np.zeros_like(output, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_tri), (255, 255, 255))
    
    # Blend warped triangle into final output
    output[np.where(mask != 0)] = warped[np.where(mask != 0)]

# Tile the texture image across the entire floor region
texture_repeats = 3
tiled_texture = np.tile(texture_image, (texture_repeats, texture_repeats, 1))
tiled_texture = cv2.resize(tiled_texture, (scene_image.shape[1], scene_image.shape[0]))

# Warp each triangle separately
for simplex in tri.simplices:
    src_tri = np.float32([
        [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
    ])[:3]  # Select 3 points from the square texture

    dst_tri = hull_points[simplex]
    
    warp_triangle(tiled_texture, src_tri, dst_tri, final_output)

# Blend with the original scene
blended_result = cv2.addWeighted(scene_image, 0.5, final_output, 0.5, 0)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB))
plt.title("Original Scene Image")

plt.subplot(1, 3, 2)
plt.imshow(depth_map, cmap='inferno')
plt.title("Estimated Depth Map")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(blended_result, cv2.COLOR_BGR2RGB))
plt.title("Final Warped Texture on Floor")

plt.show()
