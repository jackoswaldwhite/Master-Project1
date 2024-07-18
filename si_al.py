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
    r = 30  # Radius of the mask
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
    # compute the saliency map
    saliency_map = compute_saliency_map(image)

    #adaptive thresholding
    binary_mask = adaptive_thresholding(saliency_map)

    # Isolate the salient object
    mask_3_channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    salient_object = cv2.bitwise_and(image, mask_3_channel)

    #  spatial frequency filter
    spatial_freq_image = spatial_frequency_filter(salient_object)

    # combine the saliency map and the spatial frequency image
    combined_saliency = cv2.addWeighted(saliency_map, 0.5, cv2.cvtColor(spatial_freq_image, cv2.COLOR_BGR2GRAY), 0.5, 0)
    
    # normalize to ensure the final output is in the 0-255 range
    combined_saliency = cv2.normalize(combined_saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # binary thresholding for black and white output
    _, black_and_white_saliency = cv2.threshold(combined_saliency, 127, 255, cv2.THRESH_BINARY)
    
    return black_and_white_saliency

def display_and_save_result(original_image, combined_saliency_map, output_path='final_saliency_output.jpg'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(combined_saliency_map, cmap='gray')
    axs[1].set_title(' Saliency Map')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


image_path =  "C://Users//rarun//OneDrive//Documents//label3//images//2.webp"  
image = cv2.imread(image_path)

# combine the methods 
combined_saliency_map = combine_methods(image)

#  final result
display_and_save_result(image, combined_saliency_map)