# linelength_event_detector

The `linelength_event_detector` Python package. This package contains code to:

- Detect abnormal events in wave data (namely, spikes in EEG data) using line-length transform algorithm.
- Take a matrix of data (n_elecs x n_samples) and output an array of onsets and offsets of detected spikes (n_spikes x 2) 
and an array of the electrodes involved in each spike event (n_spikes x n_elecs), with them being labeled 1 for active, 0 for not.

## Package setup.
### 1. Install required packages

To install the package and have a copy of the code to edit locally, navigate to where you would like to store the package code in your terminal. Clone the package  and then install:
```
git clone https://github.com/ChangLabUcsf/linelength_event_detector.git
pip install -e linelength_event_detector
```

If you just want to use the package as is but still want to install it in your own Python environment, use the following command instead:
```
pip install git+https://github.com/Kleen-Lab/linelength_event_detector.git
```

