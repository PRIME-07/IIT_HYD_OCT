import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Image Processing Logic (wrapped in a function) 
def process_image(original_img, padding, blur_kernel_size, canny_threshold1, canny_threshold2):
    """Takes a PIL image and parameters, returns all intermediate and final images."""
    np_img = np.array(original_img.convert('L'))
    h, w = np_img.shape

    # 1. Blur - Ensure kernel size is a positive, odd integer
    blur_kernel_size = int(max(1, blur_kernel_size // 2 * 2 + 1))
    blurred_img = cv2.medianBlur(np_img, blur_kernel_size)

    # 2. Canny Edges
    edges = cv2.Canny(blurred_img, canny_threshold1, canny_threshold2)

    # 3. Find boundaries
    edge_profile = np.sum(edges, axis=1)
    significant_rows = np.where(edge_profile > (w * 0.05))[0]
    
    if len(significant_rows) > 0:
        top_boundary = significant_rows.min()
        bottom_boundary = significant_rows.max()
    else:
        top_boundary = h // 4
        bottom_boundary = h - (h // 4)

    top_crop = max(0, top_boundary - padding)
    bottom_crop = min(h, bottom_boundary + padding)

    # 4. Final Crop
    cropped_img = original_img.crop((0, top_crop, w, bottom_crop))
    
    return blurred_img, edges, cropped_img

# Main Visualization and Interaction Code 
if __name__ == '__main__':
    # 1. SET YOUR IMAGE PATH HERE
    image_path = "src/test_imgs/g/b_OS_FU_A_001010.jpg" #<-- CHANGE THIS
    
    # Initial parameters
    init_padding = 20
    init_blur = 5
    init_thresh1 = 40
    init_thresh2 = 120

    # Load the original image
    original_img = Image.open(image_path).convert("RGB")

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Adjust layout to make space for sliders
    plt.subplots_adjust(bottom=0.35)

    # Perform initial processing
    blurred_img, edges, cropped_img = process_image(
        original_img, init_padding, init_blur, init_thresh1, init_thresh2
    )

    # Initial Plotting 
    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title("1. Original Image")
    axs[0, 0].axis("off")

    im_blur = axs[0, 1].imshow(blurred_img, cmap='gray')
    axs[0, 1].set_title("2. Median Blur")
    axs[0, 1].axis("off")

    im_edges = axs[1, 0].imshow(edges, cmap='gray')
    axs[1, 0].set_title("3. Canny Edges")
    axs[1, 0].axis("off")

    im_crop = axs[1, 1].imshow(cropped_img)
    axs[1, 1].set_title("4. Final Crop")
    axs[1, 1].axis("off")
    
    # Create Slider Axes 
    ax_padding = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_blur = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_thresh1 = plt.axes([0.25, 0.10, 0.65, 0.03])
    ax_thresh2 = plt.axes([0.25, 0.05, 0.65, 0.03])

    # Create Sliders
    slider_padding = Slider(ax=ax_padding, label='Padding', valmin=0, valmax=300, valinit=init_padding, valstep=1)
    slider_blur = Slider(ax=ax_blur, label='Blur Kernel', valmin=1, valmax=21, valinit=init_blur, valstep=2)
    slider_thresh1 = Slider(ax=ax_thresh1, label='Canny Thresh 1', valmin=0, valmax=300, valinit=init_thresh1)
    slider_thresh2 = Slider(ax=ax_thresh2, label='Canny Thresh 2', valmin=0, valmax=500, valinit=init_thresh2)

    # Update Function (called when a slider is changed) 
    def update(val):
        # Get current slider values
        padding = int(slider_padding.val)
        blur = int(slider_blur.val)
        thresh1 = int(slider_thresh1.val)
        thresh2 = int(slider_thresh2.val)
        
        # Reprocess the image with new values
        new_blur, new_edges, new_crop = process_image(
            original_img, padding, blur, thresh1, thresh2
        )
        
        # Update the data in the plots
        im_blur.set_data(new_blur)
        im_edges.set_data(new_edges)
        im_crop.set_data(new_crop)
        
        # Redraw the figure
        fig.canvas.draw_idle()

    # Register the update function with each slider
    slider_padding.on_changed(update)
    slider_blur.on_changed(update)
    slider_thresh1.on_changed(update)
    slider_thresh2.on_changed(update)

    plt.show()