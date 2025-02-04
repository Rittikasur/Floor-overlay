import cv2
import numpy as np # i just wamted to get it off of my chest as i'm not sure i can speak to him aout this...because he's isnt mentally ready to be empathetic towards me when he's also hurt

def apply_floor_design(room_image_path, floor_design_path, output_image_path):
    # Load the room image and floor design
    room_image = cv2.imread(room_image_path)
    floor_design = cv2.imread(floor_design_path)
    
    # Resize the floor design to make it easier to warp
    floor_design = cv2.resize(floor_design, (500, 500))
    
    # Display instructions to the user
    print("Please select the corners of the floor area in the room image.")
    print("Click four points in clockwise order starting from the top-left.")
    
    # Define a list to store the points
    points = []

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the point
            points.append((x, y))
            # Draw a small circle on the image
            cv2.circle(temp_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Floor Area", temp_image)
            # Once we have 4 points, close the window
            if len(points) == 4:
                cv2.destroyWindow("Select Floor Area")
    
    # Create a copy of the room image for marking
    temp_image = room_image.copy()
    
    # Set up the OpenCV window and callback
    cv2.imshow("Select Floor Area", temp_image)
    cv2.setMouseCallback("Select Floor Area", select_points)
    cv2.waitKey(0)
    
    # Ensure we have exactly 4 points
    if len(points) != 4:
        print("Error: You must select exactly 4 points.")
        return
    
    # Define the destination points (corners of the floor design)
    height, width, _ = floor_design.shape
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Compute the perspective transform matrix
    src_points = np.array(points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    
    # Warp the floor design to fit the floor area
    warped_design = cv2.warpPerspective(floor_design, matrix, (room_image.shape[1], room_image.shape[0]))
    
    # Create a mask for blending
    mask = np.zeros_like(room_image, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(src_points), (255, 255, 255))
    
    # Combine the warped design with the room image
    masked_warped_design = cv2.bitwise_and(warped_design, mask)
    masked_room = cv2.bitwise_and(room_image, cv2.bitwise_not(mask))
    final_image = cv2.add(masked_warped_design, masked_room)
    
    # Save and display the result
    cv2.imwrite(output_image_path, final_image)
    cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage

room = "D:/Quleep/Prototype/Code/Data/image2.jpg"

floor = "D:/Quleep/Prototype/Code/Data/floor2.jpg"

output = "D:/Quleep/Prototype/Code/Data/output/output.jpg"

apply_floor_design(room, floor,output)
