# WORK 3: 

## Team Members

-   Antonio Arcas - [antonio.arcas\@estudiantat.upc.edu](mailto:antonio.arcas@estudiantat.upc.edu)
-   Laia Barcenilla - [laia.barcenilla\@estudiantat.upc.edu](mailto:laia.barcenilla@estudiantat.upc.edu)
-   Núria Cardona - [nuria.cardona\@estudiantat.upc.edu](mailto:nuria.cardona@estudiantat.upc.edu)
-   Ricard Garcia - [ricard.garcia\@estudiantat.upc.edu](mailto:ricard.garcia@estudiantat.upc.edu)

## Project Description

## File Structure

Here is an overview of the files and folders included in the `code` directory:

``` plaintext
CANVIAR AIXO PER L'ESTRUCTURA DEL NOU!!!!!!!
Code/
│
├── data/
│   ├── hepatitis/                
│   ├── hypothyroid/              
│
├── main.py                   
│
├── preprocessing.py              # Contains functions for data loading and preprocessing
├── helper_functions.py           # Utility functions used by other scripts
│
├── kIBLAlgorithm.py              # Core implementation of the k-IBL algorithm
├── fw_kIBLAlgorithm.py           # Core implementation of the k-IBL algorithm with feature weighting
├── ir_kIBLAlgorithm.py           # Implementation of instance reduction (IB3, MENN, MCNN)
├── svmAlgorithm.py               # Script defining and running the SVM experiments
│
├── statistical_analysis.py       # Script to perform statistical tests on results
│
└── requirements.txt              # All required Python packages     
```

## Setup and Installation

Follow these steps to set up the Python environment and install the required dependencies.

1.  **Navigate to the Code Folder** Open a terminal and change the directory to the project's `code` folder. `bash     cd <root_folder_of_project>/Code/`

2.  **Create a Virtual Environment** We recommend using a virtual environment to manage dependencies. `bash     python3 -m venv venv`

3.  **Activate the Virtual Environment**

    -   On **macOS / Linux**: `bash     source venv/bin/activate`
    -   On **Windows**: `bash     .\venv\Scripts\activate`

4.  **Install Required Dependencies** Install all packages listed in `requirements.txt`. `bash     pip install -r requirements.txt` Check if dependencies were installed by running `pip list`.

5.  **Close the Virtual Environment** `bash     deactivate`

## How to Run the Code

Make sure you have completed the previous setup steps.

1.  **Activate the Virtual Environment (if not already active)**

    -   On **macOS / Linux**: `bash     source venv/bin/activate`
    -   On **Windows**: `bash     .\venv\Scripts\activate`

2.  **Run the Main Experiment Script**

3.  **Deactivate the Virtual Environment** Once the script has finished running, you can close the virtual environment. `bash     deactivate`
