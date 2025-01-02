import cv2
import numpy as np

# Load the binary mask
mask = cv2.imread('C:/Users/datacore/Desktop/Quleep/Prototype/Code/demo.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold (if required)
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours)

# Draw contours on a copy of the mask
output = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Display the contours
cv2.imshow('Contours', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output
cv2.imwrite('contours_output.png', output)

# Print each contour's points
for contour in contours:
    print(contour)
