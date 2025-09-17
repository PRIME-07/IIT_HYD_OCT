import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Image Processing Logic using Contours
def process_image(original_img, padding, blur_kernel_size, canny_threshold1, canny_threshold2, kernel_h, kernel_w):
    np_img = np.array(original_img.convert('L'))
    h, w = np_img.shape

    blur_kernel_size = int(max(1, blur_kernel_size // 2 * 2 + 1))
    blurred_img = cv2.medianBlur(np_img, blur_kernel_size)
    edges = cv2.Canny(blurred_img, canny_threshold1, canny_threshold2)

    kernel_h = int(max(1, kernel_h // 2 * 2 + 1))
    kernel_w = int(max(1, kernel_w // 2 * 2 + 1))
    kernel = np.ones((kernel_h, kernel_w), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, cont_w, cont_h = cv2.boundingRect(largest_contour)
        top_boundary = y
        bottom_boundary = y + cont_h
        
        # Draw contour on a copy of the closed edges for visualization
        contour_viz = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_viz, [largest_contour], -1, (0,255,0), 2)
    else:
        top_boundary = h // 4
        bottom_boundary = h - (h // 4)
        contour_viz = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR)


    top_crop = max(0, top_boundary - padding)
    bottom_crop = min(h, bottom_boundary + padding)
    cropped_img = original_img.crop((0, top_crop, w, bottom_crop))
    
    return blurred_img, contour_viz, cropped_img

# Main Visualization and Interaction Code
if __name__ == '__main__':
    image_path = "src/test_imgs/g/b_OS_FU_A_001010.jpg"
    
    init_padding = 0
    init_blur = 5
    init_thresh1 = 40
    init_thresh2 = 120
    init_kernel_h = 5
    init_kernel_w = 11

    original_img = Image.open(image_path).convert("RGB")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.45)

    blurred_img, contour_viz, cropped_img = process_image(
        original_img, init_padding, init_blur, init_thresh1, init_thresh2, init_kernel_h, init_kernel_w
    )

    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title("1. Original Image")
    axs[0, 0].axis("off")

    im_blur = axs[0, 1].imshow(blurred_img, cmap='gray')
    axs[0, 1].set_title("2. Median Blur")
    axs[0, 1].axis("off")

    im_contour = axs[1, 0].imshow(contour_viz)
    axs[1, 0].set_title("3. Closed Edges + Largest Contour")
    axs[1, 0].axis("off")

    im_crop = axs[1, 1].imshow(cropped_img)
    axs[1, 1].set_title("4. Final Crop")
    axs[1, 1].axis("off")
    
    ax_padding = plt.axes([0.25, 0.30, 0.65, 0.03])
    ax_blur = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_thresh1 = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_thresh2 = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_kernel_h = plt.axes([0.25, 0.10, 0.65, 0.03])
    ax_kernel_w = plt.axes([0.25, 0.05, 0.65, 0.03])

    slider_padding = Slider(ax=ax_padding, label='Padding', valmin=0, valmax=100, valinit=init_padding, valstep=1)
    slider_blur = Slider(ax=ax_blur, label='Blur Kernel', valmin=1, valmax=21, valinit=init_blur, valstep=2)
    slider_thresh1 = Slider(ax=ax_thresh1, label='Canny Thresh 1', valmin=0, valmax=255, valinit=init_thresh1)
    slider_thresh2 = Slider(ax=ax_thresh2, label='Canny Thresh 2', valmin=0, valmax=500, valinit=init_thresh2)
    slider_kernel_h = Slider(ax=ax_kernel_h, label='Kernel Height', valmin=1, valmax=21, valinit=init_kernel_h, valstep=2)
    slider_kernel_w = Slider(ax=ax_kernel_w, label='Kernel Width', valmin=1, valmax=41, valinit=init_kernel_w, valstep=2)

    def update(val):
        new_blur, new_contour_viz, new_crop = process_image(
            original_img, slider_padding.val, slider_blur.val, slider_thresh1.val, slider_thresh2.val,
            slider_kernel_h.val, slider_kernel_w.val
        )
        im_blur.set_data(new_blur)
        im_contour.set_data(new_contour_viz)
        im_crop.set_data(new_crop)
        fig.canvas.draw_idle()

    for s in [slider_padding, slider_blur, slider_thresh1, slider_thresh2, slider_kernel_h, slider_kernel_w]:
        s.on_changed(update)

    plt.show()