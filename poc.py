#pip install opencv-python torch torchvision pillow matplotlib

#read and preprocess iamge
import cv2
import torch

# Load image
image = cv2.imread('C:/Users/datacore/Desktop/Quleep/Prototype/Data/image1.jpg')

# Resize if necessary
resized_image = cv2.resize(image, (512, 512))  # Resize to a manageable size
cv2.imshow("Original Room Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




#Floor Segmentation
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load DeepLabV3 Model
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Preprocess image for the model
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(resized_image).unsqueeze(0)

# Pass through the model
with torch.no_grad():
    output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

# Visualize segmentation result
floor_mask = (output_predictions == 15).astype(np.uint8)  # Floor class in COCO dataset is labeled as 15
floor_segmented = cv2.bitwise_and(resized_image, resized_image, mask=floor_mask)

cv2.imshow("Floor Segmentation", floor_segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()




#Depth Estimation
import torch.hub

# Load pre-trained MiDaS depth model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to('cpu')
midas.eval()

# Transform input for MiDaS
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

input_batch = transform(resized_image).unsqueeze(0)

# Predict depth
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),
                                                 size=resized_image.shape[:2],
                                                 mode="bicubic",
                                                 align_corners=False).squeeze()
    depth_map = prediction.cpu().numpy()

# Normalize depth map
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("Depth Map", depth_map_normalized.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()



#Carpet Placement

# Load carpet texture
carpet_texture = cv2.imread('C:/Users/datacore/Desktop/Quleep/Prototype/Data/floor1.jpg')

# Resize the texture to fit the floor area
carpet_texture_resized = cv2.resize(carpet_texture, (resized_image.shape[1], resized_image.shape[0]))

# Apply carpet texture only on the segmented floor area
carpet_on_floor = np.where(floor_mask[:, :, None] == 1, carpet_texture_resized, resized_image)

cv2.imshow("Carpet Placed", carpet_on_floor)
cv2.waitKey(0)
cv2.destroyAllWindows()
#




