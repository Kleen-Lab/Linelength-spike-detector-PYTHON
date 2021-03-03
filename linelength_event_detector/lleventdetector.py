import numpy as np

def lleventdetector(L, sfx, prc, mel):
    """Detects events (aka spikes) in data and identifies start/stop times and channels involved.

        Takes 1d/2d list (L) of linelength transformed data and finds segments
        surpassing a percentile threshold. Returns start & stop times of detected
        event and channels involved in the form of 2d lists. Automatically deletes
        events that are considered too short (<mel).

        Parameters
        ----------
        L : list of float values
            line-length transform values for each data point (returned from lltransform.py)
        sfx : int
            sampling frequency of data
        prc : float
            percentile used to determine detection threshold
        mel : int
            "minimum event length" (in ms)
            minimum acceptable length of event in milliseconds
            event will be deleted from detection if it is shorter
            than this length

        Returns
        -------
        ets : 2d list of start and stop times of detected events
            each row is 1 event
            column 0 is start times
            column 1 is stop times
        ech : 1d list of strings
            each string is 1 event
            each number in each string is a channel involved in that event

        Raises
        ------
        RuntimeError
            If the amount of start/stop times are unequal, meaning the timestamps of events are misaligned.
        """

    llw = 0.04  # default value of line-length window

    # transpose L if needed
    if len(L) < len(L[0]):
        L = np.transpose(L)
        flipped = True
    else:
        flipped = False

    Lvec = L.flatten()
    Lvec[np.isnan(Lvec)] = 0
    thresh = np.percentile(Lvec, prc)  # calculate threshold

    # populate with 1s where values exceed threshold
    L[np.isnan(L)] = 0
    Li = L > thresh
    Li = 1 * Li

    # consolidate (sum of index 1's) across channels
    a = np.nansum(Li, 1)

    # index 1 when at least 1 channel exceeds threshold
    a = (a > 0) * 1

    # find event start/stop times
    a = np.diff(a)
    eON = np.where(a == 1)[0]
    eOFF = np.where(a == -1)[0]

    # Edge Case 1: If above threshold prior to beginning of data,
    #               create an eON value at first sample ('0')
    if len(eOFF) > len(eON):
        eON = np.insert(eON, 0, 0)

    # Edge Case 2: If above threshold after end of data,
    #               create an eOFF value at last sample ('len(Li)-1')
    if len(eOFF) < len(eON):
        eOFF = np.append(eOFF, len(Li) - 1)

    # Raise error if eON and eOFF are different lengths for some reason
    if len(eOFF) != len(eON):
        raise RuntimeError('eON and eOFF are different lengths, check your code.')

    ets = np.column_stack((eON, eOFF))

    # flip L back if transposed previously
    if flipped:
        L = np.transpose(L)

    # index channels involved in each event
    ech = np.zeros((len(ets), len(Li[0])), dtype=int)
    for i in range(len(ets)):
        on_event = ets[i, 0]
        off_event = ets[i, 1]
        total_event = Li[on_event:off_event]
        ech[i] = (np.nansum(total_event, 0) >= 1) * 1

    # delete events that are shorter than 'mel'
    # convert mel to number of samples (rounded)
    numsamples = np.round((sfx * mel)/1000)
    delete = []

    for x in range(len(ets)):
        on_event = ets[x, 0]
        off_event = ets[x, 1]
        if off_event - on_event < numsamples:
            delete.append(x)

    ets = np.delete(ets, delete, 0)
    ech = np.delete(ech, delete, 0)

    # add modality that creates a list of channels involved in each event as a string list
    channels = []
    ch_str = ''
    for p in range(len(ech)):  # for each event
        for q in range(len(ech[0])):  # for each channel in event
            if ech[p][q] == 1:
                ch_str += str(q) + ','
        channels.append(ch_str)
        ch_str = ''

    ech = channels

    # Center LL transform window
    ets = (np.round(ets + (sfx * llw) / 2))

    return ets, ech
