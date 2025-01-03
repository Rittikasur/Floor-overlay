import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

def load_pretrained_model():
    """Load a pretrained DeepLabV3 model for semantic segmentation."""
    model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)
    return model

def apply_floor_design_using_dl(room_image_path, floor_design_path, output_image_path):
    # Load the room image and floor design
    room_image = cv2.imread(room_image_path)
    floor_design = cv2.imread(floor_design_path)

    # Resize the floor design to a fixed size
    floor_design = cv2.resize(floor_design, (500, 500))

    # Prepare the room image for prediction (resize and preprocess)
    image_for_model = cv2.resize(room_image, (256, 256))  # Resize for the model input
    image_for_model = tf.convert_to_tensor(image_for_model, dtype=tf.float32)
    image_for_model = tf.expand_dims(image_for_model, axis=0)  # Add batch dimension

    # Load the pre-trained semantic segmentation model
    model = load_pretrained_model()

    # Predict the segmentation mask for the room image
    predictions = model.predict(image_for_model)  # Shape (1, 256, 256, 21) for 21 classes in output
    floor_mask = predictions[0, :, :, 0]  # Assuming '0' is the class for the floor

    # Threshold the mask to create a binary floor mask
    floor_mask = np.where(floor_mask > 0.5, 255, 0).astype(np.uint8)

    # Resize the mask back to the original room image size
    floor_mask_resized = cv2.resize(floor_mask, (room_image.shape[1], room_image.shape[0]))

    # Find contours in the mask to identify the floor area
    contours, _ = cv2.findContours(floor_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (likely to be the floor)
    floor_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to get a polygon
    epsilon = 0.05 * cv2.arcLength(floor_contour, True)
    approx = cv2.approxPolyDP(floor_contour, epsilon, True)

    # Get the width and height of the floor design
    height, width, _ = floor_design.shape

    # Define destination points for the perspective transform
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # Apply the perspective transform if enough points are detected
    if len(approx) >= 4:
        src_points = np.array([point[0] for point in approx], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        warped_design = cv2.warpPerspective(floor_design, matrix, (room_image.shape[1], room_image.shape[0]))

        # Create a mask for blending the warped floor design
        mask = np.zeros_like(room_image, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(src_points), (255, 255, 255))

        # Combine the warped floor design with the room image
        masked_warped_design = cv2.bitwise_and(warped_design, mask)
        masked_room = cv2.bitwise_and(room_image, cv2.bitwise_not(mask))
        final_image = cv2.add(masked_warped_design, masked_room)

        # Save and display the result
        cv2.imwrite(output_image_path, final_image)
        cv2.imshow("Final Image", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
apply_floor_design_using_dl("room_image.jpg", "floor_design.jpg", "output.jpg")
