# Linelength-spike-detector-PYTHON

The `Linelength-spike-detector-PYTHON` Python package. This package contains code to:

- Detect abnormal events in wave data (namely, spikes in EEG data) using linelength transform algorithm.
- Take a matrix of data (n_elecs x n_samples) and output an array of onsets and offsets of detected spikes (n_spikes x 2) 
and an array of the electrodes involved in each spike event (n_spikes x n_elecs), with them being labeled 1 for active, 0 for not.
- Note on terminology: words "spike" and "event" are used interchangeably in function/variable names for the purpose of this code. 

Based on Estellar et al 2001, DOI 10.1109/IEMBS.2001.1020545

## Package setup.
### 1. Install required packages

To install the package and have a copy of the code to edit locally, navigate to where you would like to store the package code in your terminal. Clone the package  and then install:
```
git clone https://github.com/Kleen-Lab/Linelength-spike-detector-PYTHON.git
pip install -e Linelength-spike-detector-PYTHON
```

If you just want to use the package as is but still want to install it in your own Python environment, use the following command instead:
```
pip install git+https://github.com/Kleen-Lab/Linelength-spike-detector-PYTHON
```

