import cv2
import numpy as np
from matplotlib.font_manager import FontProperties

# Load the image using OpenCV
image_path = "/file_system/vepfs/algorithm/dujun.nie/code/WMNav/VLMnav/logs/ObjectNav_version_7_pro_improve_reset/4_of_50/83_890/step0/color_sensor.png"
img = cv2.imread(image_path)  # Read the image

# Get the image dimensions
height, width, _ = img.shape

# Calculate the width and height of each grid cell
grid_width = width // 3
grid_height = height // 3

# Set the font properties and size (using a font path that exists in your system)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.5  # You can adjust this to control the font size
font_thickness = 3
font_color = (230, 216, 173)  # Light blue color in BGR format

# Draw the grid lines and numbers
index = 1
for i in range(1, 3):
    # Draw vertical and horizontal lines (grid lines)
    cv2.line(img, (i * grid_width, 0), (i * grid_width, height), (255, 255, 255), 3)  # Vertical line
    cv2.line(img, (0, i * grid_height), (width, i * grid_height), (255, 255, 255), 3)  # Horizontal line

for i in range(3):
    for j in range(3):
        x = j * grid_width
        y = i * grid_height
        text = str(index)

        # Get the text size to center it within the cell
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_size, font_thickness)

        # Calculate text position (to center it)
        text_x = x + (grid_width - text_width) // 2
        text_y = y + (grid_height + text_height) // 2

        # Add the text (label) to the image
        cv2.putText(img, text, (text_x, text_y), font, font_size, font_color, font_thickness)

        index += 1

# Save the result image
output_image_path = '/file_system/vepfs/algorithm/dujun.nie/output_image_with_grid_openCV.png'
cv2.imwrite(output_image_path, img)

# Optionally, display the image
# cv2.imshow('Image with Grid and Labels', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
