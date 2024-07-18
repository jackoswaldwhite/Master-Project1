import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def compute_saliency_map(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    blurred_image = gaussian_filter(lab_image, sigma=2)
    saliency_map = np.linalg.norm(lab_image - blurred_image, axis=2)
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return saliency_map

def adaptive_thresholding(saliency_map):
    mean_saliency = np.mean(saliency_map)
    adaptive_thresh_value = 2 * mean_saliency
    _, binary_mask = cv2.threshold(saliency_map, adaptive_thresh_value, 255, cv2.THRESH_BINARY)
    return binary_mask

def spatial_frequency_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 30  
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) * 2 + (y - center[1]) * 2 <= r*r
    mask[mask_area] = 1

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    spatial_frequency_image = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
    return spatial_frequency_image

def combine_methods(image):
    saliency_map = compute_saliency_map(image)
    binary_mask = adaptive_thresholding(saliency_map)
    mask_3_channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    salient_object = cv2.bitwise_and(image, mask_3_channel)
    spatial_freq_image = spatial_frequency_filter(salient_object)
    combined_saliency = cv2.addWeighted(saliency_map, 0.5, cv2.cvtColor(spatial_freq_image, cv2.COLOR_BGR2GRAY), 0.5, 0)
    combined_saliency = cv2.normalize(combined_saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, black_and_white_saliency = cv2.threshold(combined_saliency, 127, 255, cv2.THRESH_BINARY)
    return black_and_white_saliency

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
        distance = np.sqrt((x - prev_position[0]) * 2 + (y - prev_position[1]) * 2)
        cost = saliency_sum + lambda_val * distance

        if cost < min_cost:
            min_cost = cost
            optimal_position = (x, y)

    return optimal_position

def overlay_labels(image, labels, positions, background_opacity=0.5, text_opacity=1.0, border_color=(0, 255, 0)):
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    for label, position in zip(labels, positions):
        x, y = position
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_w, text_h = text_size

        background_color = (200, 200, 200)
        cv2.rectangle(overlay, (x, y), (x + text_w + 20, y + text_h + 20), background_color, -1)
        sub_img = overlay[y:y+text_h+20, x:x+text_w+20]
        overlay[y:y+text_h+20, x:x+text_w+20] = cv2.addWeighted(sub_img, background_opacity, sub_img, 1.0 - background_opacity, 0)
        cv2.rectangle(overlay, (x, y), (x + text_w + 20, y + text_h + 20), border_color, 2)
        font_color = (0, 255, 0)
        cv2.putText(overlay, label, (x + 10, y + text_h + 10), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

    return overlay

def main():
    cap = cv2.VideoCapture(0)

    prev_positions = {
        'Arun': (0, 0),
        'Bismarck': (0, 50)
    }

    lambda_val = 0.1
    sample_rate = 30
    overlay_size = (50, 50)
    smoothing_factor = 0.5

    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        combined_saliency_map = combine_methods(frame)
        labeled_positions = {}

        for label, prev_position in prev_positions.items():
            optimal_position = find_optimal_label_position(combined_saliency_map, overlay_size, prev_position, lambda_val, sample_rate)
            prev_positions[label] = (int(prev_position[0] * (1 - smoothing_factor) + optimal_position[0] * smoothing_factor),
                                     int(prev_position[1] * (1 - smoothing_factor) + optimal_position[1] * smoothing_factor))
            labeled_positions[label] = prev_positions[label]

        background_opacity = 0.3
        text_opacity = 1.0
        border_color = (0, 200, 0)

        labeled_frame = overlay_labels(frame, list(prev_positions.keys()), list(labeled_positions.values()), background_opacity, text_opacity, border_color)

        ax[0].clear()
        ax[0].imshow(cv2.cvtColor(labeled_frame, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Labeled Frame')
        ax[0].axis('off')

        ax[1].clear()
        ax[1].imshow(combined_saliency_map, cmap='gray')
        ax[1].set_title('Saliency Map')
        ax[1].axis('off')

        plt.draw()
        plt.pause(0.001)

        if plt.waitforbuttonpress(0.001):
            break

    cap.release()
    plt.close()

if __name__ == "__main__":
    main()