import cv2
import numpy as np

def achanta_saliency(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2.5)
    saliency = cv2.absdiff(gray, blurred)
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    return saliency

def find_optimal_label_position(saliency_map, overlay_size, prev_position, lambda_val, sample_rate):
    frame_height, frame_width = saliency_map.shape
    overlay_height, overlay_width = overlay_size
    sampled_pixels = [(x, y) for x in range(0, frame_width - overlay_width, sample_rate)
                      for y in range(0, frame_height - overlay_height, sample_rate)]
    
    min_cost = float('inf')
    optimal_position = prev_position
    
    for (x, y) in sampled_pixels:
        region = saliency_map[y:y+overlay_height, x:x+overlay_width]
        saliency_sum = np.sum(region)
        distance = np.sqrt((x - prev_position[0]) ** 2 + (y - prev_position[1]) ** 2)
        cost = saliency_sum + lambda_val * distance
        
        if cost < min_cost:
            min_cost = cost
            optimal_position = (x, y)
    
    return optimal_position

def overlay_labels(image, label, position):
    x, y = position
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)
    font_thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_w, text_h = text_size
    
    sub_img = overlay[y:y+text_h+20, x:x+text_w+20]
    background = cv2.GaussianBlur(sub_img, (21, 21), 0)
    overlay[y:y+text_h+20, x:x+text_w+20] = background

    cv2.addWeighted(background, 0.6, overlay[y:y+text_h+20, x:x+text_w+20], 0.4, 0, overlay[y:y+text_h+20, x:x+text_w+20])
    cv2.rectangle(overlay, (x, y), (x+text_w+20, y+text_h+20), (0, 255, 0), 2)
    cv2.putText(overlay, label, (x + 10, y + text_h + 10), font, font_scale, font_color, font_thickness)

    return overlay

def main():
    cap = cv2.VideoCapture(0)
    prev_position_arun = (0, 0)
    prev_position_bismarck = (0, 50)
    lambda_val = 0.1
    sample_rate = 30
    overlay_size = (50, 50)
    smoothing_factor = 0.5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        saliency_map = achanta_saliency(frame)
        optimal_position_arun = find_optimal_label_position(saliency_map, overlay_size, prev_position_arun, lambda_val, sample_rate)
        optimal_position_bismarck = find_optimal_label_position(saliency_map, overlay_size, prev_position_bismarck, lambda_val, sample_rate)
        
        prev_position_arun = (int(prev_position_arun[0] * (1 - smoothing_factor) + optimal_position_arun[0] * smoothing_factor),
                              int(prev_position_arun[1] * (1 - smoothing_factor) + optimal_position_arun[1] * smoothing_factor))
        prev_position_bismarck = (int(prev_position_bismarck[0] * (1 - smoothing_factor) + optimal_position_bismarck[0] * smoothing_factor),
                                  int(prev_position_bismarck[1] * (1 - smoothing_factor) + optimal_position_bismarck[1] * smoothing_factor))
        
        labeled_frame = overlay_labels(frame, 'Arun', prev_position_arun)
        labeled_frame = overlay_labels(labeled_frame, 'Bismarck', prev_position_bismarck)
        
        cv2.imshow('AR Label Placement', labeled_frame)
        cv2.imshow('Saliency Map', saliency_map)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
