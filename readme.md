# IIT_HYD_OCT Project - [Your Branch Name Here]

## Overview

This repository branch contains a collection of scripts for **Optical Coherence Tomography (OCT)** image analysis. This branch contains a set of scripts designed for testing classification models and testing/refining image quality assessment metrics.

### Key Features:

* **Average_mask.ipynb**: This script uses **heuristics** and existing **Image Quality Assessment (IQA)** data to visualize segmentable mask regions. Its goal is to identify and show these regions without relying on reference images for traditional metrics like **Signal-to-Noise Ratio (SNR)** or **Peak Signal-to-Noise Ratio (PSNR)**.

* **Band_Straightening.ipynb**: A prototype script for `Average_mask`. It straightens the image to create a uniform band across the choroid, which helps in passing localized regions to compute the same metrics as present in Average_mask.

* **Classifier.ipynb**: This module trains a classifier using both the **segmentation mask** and the **original image**. The purpose is to evaluate the quality of the segmentation by how well the classifier performs.

* **Classifier_Mask_only.ipynb**: Similar to the above, but this classifier uses **only the segmentation mask** as input. This approach helps to isolate and evaluate the diagnostic value of the structural information contained within the mask itself.

* **IQA_Metrics.ipynb**: This script is dedicated to calculating and evaluating **no-reference IQA metrics**, which assess image quality without the need for a reference image.

---

## Setup

### Installation

1.  **Clone this repository**:
    ```bash
    git clone [https://github.com/PRIME-07/IIT_HYD_OCT.git](https://github.com/PRIME-07/IIT_HYD_OCT.git)
    ```

2.  **Navigate to the project directory**:
    ```bash
    cd IIT_HYD_OCT
    ```

3.  **Switch to this branch**:
    ```bash
    git checkout [Your Branch Name Here]
    ```

4.  **Create and activate a virtual environment** (optional but highly recommended):
    ```bash
    # Create the environment
    python -m venv venv
    # Activate on Linux/macOS
    source venv/bin/activate
    # Activate on Windows
    venv\Scripts\activate
    ```

5.  **Install the necessary dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Executing Notebooks

The primary way to use this branch is by running the provided Jupyter Notebooks.

* **Jupyter/VS Code**: Open the notebooks directly using Jupyter Notebook or the VS Code Notebook interface.
* **File Paths**: **Before execution, ensure you update all file paths within the notebooks** to match your local environment.
* **Execution Order**: To reproduce the results, execute the cells in each notebook sequentially.

### Converting to Python Scripts

If you need to run the notebooks from the command line or integrate them into a larger pipeline, you can convert them to Python scripts.

* **Command**: Use `jupyter nbconvert` to convert a notebook to a `.py` file.
    ```bash
    jupyter nbconvert --to script filename.ipynb
    ```

### Outputs

* All outputs, including saved models and analysis results, are stored in dedicated folders as specified in each notebook's code.

---

## Notes

* **Input Data**: Make sure all input images are placed in the correct directories as expected by the scripts.
* **Environment**: These notebooks were tested with **Python 3.10** and **Torch 2.1.2+rocm6.1.3**.
* **Dependencies**: For the IQA metrics, the `pyiqa` library is a required dependency.
