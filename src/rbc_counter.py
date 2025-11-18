import cv2
import numpy as np
import os
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict

# --- Configuration ---
# Update this path to where your BCCD images are stored
DATA_DIR = 'data/raw/'
RESULTS_DIR = 'results/processed_images/'
METRICS_PATH = 'results/metrics/evaluation_summary.csv'

class RBCCounter:
    """
    Implements the RBC detection and counting pipeline using traditional
    Digital Image Processing (DIP) techniques.
    """

    def __init__(self, output_dir: str = RESULTS_DIR):
        """
        Initializes the counter with the output directory.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"RBC Counter initialized. Results will be saved to: {self.output_dir}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Preprocessing: Gaussian Filtering and Histogram Equalization.
        
        Args:
            image: The input microscopy image (BGR format).
        
        Returns:
            The enhanced grayscale image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Gaussian Filtering for noise reduction
        # Using a small kernel (e.g., 3x3 or 5x5) to smooth but retain edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Histogram Equalization for contrast enhancement
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) is often better
        # than standard histEq for local contrast.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced

    def detect_rbc_watershed(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Implements the Watershed Segmentation method for RBC counting,
        which is ideal for separating touching/overlapping cells.

        Args:
            image: The preprocessed grayscale image.

        Returns:
            A tuple: (count, annotated_image)
        """
        print("   -> Running Watershed Segmentation...")
        
        # 1. Thresholding to create a binary mask (Otsu's method)
        # We assume the cells are brighter than the background.
        # cv2.THRESH_BINARY_INV might be needed if cells are dark.
        # Experimentation required based on dataset.
        _, thresh = cv2.threshold(image, 0, 255, 
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Morphological operations: Noise removal/Smoothing
        kernel = np.ones((3, 3), np.uint8)
        # Opening: Erosion followed by Dilation (removes small objects)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # 3. Find sure background area
        # Dilation expands the foreground, leaving only clear background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 4. Find sure foreground area (the cell 'centers')
        # Distance Transform calculates the distance to the closest background pixel
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # Threshold the distance transform to get definite foreground markers
        # The threshold value (e.g., 0.6*max) determines the minimum distance 
        # for a point to be considered a center.
        ret, sure_fg = cv2.threshold(dist_transform, 
                                     0.6 * dist_transform.max(), 255, 0)
        
        # 5. Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 6. Label the foreground markers
        # Markers are the starting points for the watershed algorithm
        markers = ndi.label(sure_fg)[0]
        # Add 1 to all labels so that sure background is 1 instead of 0
        markers = markers + 1
        # Now, mark the unknown region with 0
        markers[unknown == 255] = 0

        # 7. Apply Watershed
        # The image must be BGR/RGB for the watershed function in OpenCV/skimage
        # We run the watershed on the grayscale image, using the markers
        labels = watershed(-dist_transform, markers, mask=opening)
        
        # Count the number of unique cell labels (excluding the background label 1)
        # Unique labels are the distinct segmented cells.
        cell_labels = np.unique(labels)
        # The count excludes the 'background' label (1) and the 'unknown' region (0, if it was used)
        # Since we labeled background as 1, the count is number of unique labels - 1 (for label 1)
        rbc_count = len(cell_labels) - 1 

        # 8. Visualization (Annotated Image)
        annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # Convert back to color for overlay
        # Draw contours for visualization
        for label in cell_labels:
            if label == 1: # Skip background
                continue
            
            # Create a mask for the current cell
            mask = np.zeros(labels.shape, dtype="uint8")
            mask[labels == label] = 255
            
            # Find the largest contour in the mask
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size/shape (Post-processing step)
            # You would add logic here to filter out non-RBC shapes or too-small/large objects
            
            # Draw the contour on the original image
            if contours:
                cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), 2) # Green contours
        
        # Put the count text on the image
        cv2.putText(annotated_image, f"RBC Count (Watershed): {rbc_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return rbc_count, annotated_image

    # Placeholder for other methods (as outlined in your proposal)
    def detect_rbc_hough(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """Circular Hough Transform detection (WIP)."""
        # ... Implementation using cv2.HoughCircles ...
        return 0, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def detect_rbc_contour_analysis(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """Contour Analysis detection (WIP)."""
        # ... Implementation using cv2.findContours and contour properties (area, circularity) ...
        return 0, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def evaluate_dataset(data_path: str = DATA_DIR, 
                     results_path: str = RESULTS_DIR,
                     metrics_file: str = METRICS_PATH):
    """
    Main function to run the detection pipeline on a dataset, save results,
    and simulate evaluation.
    """
    print("--- Starting RBC Counting and Evaluation Pipeline ---")
    
    # 1. Initialize the Counter and Setup
    counter = RBCCounter(results_path)
    image_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Error: No images found in {data_path}. Please place image files there.")
        return

    all_results = []
    
    # Placeholder for ground truth data (you will load/parse this from files)
    # For now, we use a simple mock structure.
    # In reality, you'd load bounding boxes or coordinates for each cell.
    mock_ground_truth = {
        # 'filename.jpg': expected_count
        'image01.jpg': 100,
        'image02.jpg': 125,
        # ... add more actual ground truth counts
    }


    # 2. Process each image
    for filename in image_files:
        file_path = os.path.join(data_path, filename)
        original_bgr = cv2.imread(file_path)
        
        if original_bgr is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        print(f"\nProcessing Image: {filename}")
        
        # A. Preprocessing
        enhanced_gray = counter.preprocess_image(original_bgr)

        # B. Detection (Watershed method)
        rbc_count, annotated_image = counter.detect_rbc_watershed(enhanced_gray)

        # C. Save Results
        output_filename = f"watershed_detected_{filename}"
        cv2.imwrite(os.path.join(results_path, output_filename), annotated_image)
        print(f"   -> Count: {rbc_count}. Annotated image saved as {output_filename}")
        
        # D. Evaluation (Simulated)
        expected_count = mock_ground_truth.get(filename, np.nan)
        
        # Calculate a simple accuracy metric (replace with true metrics later)
        if not np.isnan(expected_count):
            abs_error = abs(rbc_count - expected_count)
            percent_error = (abs_error / expected_count) * 100 if expected_count > 0 else np.nan
        else:
            abs_error = np.nan
            percent_error = np.nan
        
        all_results.append({
            'filename': filename,
            'method': 'Watershed',
            'detected_count': rbc_count,
            'ground_truth_count': expected_count,
            'absolute_error': abs_error,
            'percent_error': percent_error
            # Add placeholders for Precision, Recall, F1-Score, Processing_Time
        })

    # 3. Save Metrics Summary
    results_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    results_df.to_csv(metrics_file, index=False)
    
    print("\n--- Evaluation Summary ---")
    print(results_df.head())
    print(f"\nSummary saved to: {metrics_file}")
    
    # 4. Final step: Run the main function
if __name__ == '__main__':
    # Create mock files to ensure the script runs the first time without errors
    # NOTE: You should replace these with actual BCCD images for real results
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")
    
    # Create a small dummy image for testing the pipeline structure
    # This is a highly simplified mock RBC image (white circles on black)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (25, 25), 10, (255, 255, 255), -1)
    cv2.circle(dummy_img, (50, 50), 12, (255, 255, 255), -1)
    cv2.circle(dummy_img, (75, 75), 11, (255, 255, 255), -1)
    # Two overlapping circles to test watershed principles
    cv2.circle(dummy_img, (30, 70), 8, (255, 255, 255), -1)
    cv2.circle(dummy_img, (40, 65), 8, (255, 255, 255), -1) 
    cv2.imwrite(os.path.join(DATA_DIR, 'image01.jpg'), dummy_img)
    
    evaluate_dataset()