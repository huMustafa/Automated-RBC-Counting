Automated Red Blood Cell Counter (RBC-Count-DIP)

This project implements an automated red blood cell (RBC) counting system using traditional Digital Image Processing (DIP) techniques, as outlined in the project proposal.

1. Project Structure

The project is organized as follows:

RBC-Count-DIP/
├── data/
│   ├── raw/                  # Original BCCD images (e.g., blood_smear_01.jpg)
│   └── ground_truth/         # Annotation files (e.g., CSV, XML) for evaluation
├── results/
│   ├── processed_images/     # Output images with detected cell contours
│   └── metrics/              # CSV files with performance metrics (accuracy, precision, recall)
├── src/
│   ├── __init__.py
│   └── rbc_counter.py        # Core implementation of the RBC counting pipeline (Watershed, Hough, Contour)
├── .gitignore
├── requirements.txt          # Python dependencies
└── README.md                 # This file


2. Setup and Installation

Clone the repository:

git clone <your-repo-link>
cd RBC-Count-DIP


Create a virtual environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate


Install dependencies:
The project relies on OpenCV, scikit-image, and NumPy.

# Install dependencies listed in requirements.txt
pip install -r requirements.txt


3. Usage

To run the main evaluation pipeline:

python src/rbc_counter.py
