
SiLM Package Overview
This repository contains essential components for running and analyzing Structured Illumination Localization Microscopy (SiLM) experiments. The project is organized into the following three main folders:

📁 1. CUDA Source Codes
Source code for generating the GPU DLL functions in SiLM, enabling high-performance computations.

📁 2. SiLM Analysis Pipeline
The core analysis pipeline for SiLM. Raw data for testing and analysis can be found at:
https://zenodo.org/records/15487082.

📁 3. Public SiLM
A graphical user interface (GUI) designed for:

Calculating theoretical intensity curves.

Computing the Cramér–Rao Lower Bound (CRLB) for localization precision in SiLM.

📚 Manuals
Manuals for using the SiLM analysis pipeline and CUDA code are available in each folder.

Hardware Requirements
The code has been tested on a standard computer with CUDA 12.1 installed and properly configured. A GPU that supports CUDA 12.1 is required for optimal performance.

Software Requirements
Operating System
Windows (Tested on Windows OS)

Python Dependencies
The package depends on the following Python libraries. The versions used in this project are specified below:

Numpy: Version 1.26.4

PIL: Version 10.3.0

Matplotlib: Version 3.8.4

Cv2 (OpenCV): Version 4.8.1

Tkinter: (Built-in, no specific version required)

ctypes: Version 1.1.0

Scipy: Version 1.13.0

Numba: Version 0.59.1

Tifffile: Version 2023.4.12

PyQt5: (GUI support)

Pandas

Tqdm: (Progress bar library)

Make sure to install the necessary dependencies to ensure the pipeline works as expected.

