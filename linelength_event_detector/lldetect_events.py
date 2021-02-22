from linelength_event_detector.lleventdetector import lleventdetector
from linelength_event_detector.lltransform import lltransform


def lldetect_events(d, sfx):
    """Performs line-length transform on data, then detects events in data and identifies start/stop times and channels
       involved.

               Takes 1d/2d list (L) of data, performs line-length transform on it and detects events on transformed data.
               Returns start & stop times of detected events and channels involved in the form of 2d lists.

               Parameters
               ----------
               d : list of float values
                   EEG data to be detected for events
               sfx : int
                   sampling frequency of data

               Returns
               -------
               ets : 2d list of start and stop times of detected events
                   each row is 1 event
                   column 0 is start times
                   column 1 is stop times
               ech : 1d list of strings which hold channel #s involved in event
                   each string is 1 event
                   each element in each string is the number of the channel involved
               """
    transformed_data = lltransform(d, sfx)
    prc = 99.9  # percentile needed for lleventdetector.py
    mel = 100  # minimum event length in milliseconds, needed for lleventdetector.py
    ets, ech = lleventdetector(transformed_data, sfx, prc, mel)
    return ets, ech
