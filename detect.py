import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import os
# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use yolov8n.pt or any other available YOLOv8 model

# Load the image
image_path = './Images/central-bus-stop.webp'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform detection
results = model(image_rgb)

# Create a copy of the image to draw the text boxes on
overlay = image_rgb.copy()

# Iterate over the detections and overlay text at the bottom right of each bounding box
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
    confidences = result.boxes.conf.cpu().numpy()  # Extract confidences
    class_ids = result.boxes.cls.cpu().numpy()  # Extract class IDs

    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        # label = f'{model.names[int(class_id)]}: {conf:.2f}'
        label = f'{model.names[int(class_id)]}'

        # Calculate the position for the text (bottom right of the bounding box)
        text_position = (x2, y1)

        # Calculate text size
        font_scale = 0.7
        font_thickness = 2
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_width, text_height = text_size

        # Draw a filled rectangle with a translucent effect
        box_coords = ((x2, y2), (x2 + text_width, y2 - text_height - 10))
        cv2.rectangle(overlay, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)

        # Put the text on top of the rectangle
        cv2.putText(overlay, label, (x2, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

# Blend the overlay with the original image
alpha = 0.6  # Transparency factor
image_rgb = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

# Convert back to BGR for displaying with OpenCV
image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# Display the final image
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Construct the output path to follow a similar data structure to the input folder
input_dir, input_filename = os.path.split(image_path)
output_filename = 'output_' + input_filename
output_path = os.path.join(input_dir, output_filename)

# Save the result
cv2.imwrite(output_path, image_bgr)

print(f"Output saved to: {output_path}")
