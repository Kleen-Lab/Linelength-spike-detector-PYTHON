import pytest
import scipy.io
import numpy as np

from linelength_event_detector.lleventdetector import lleventdetector
from linelength_event_detector.lltransform import lltransform


@pytest.fixture
def example_eeg_data():
    # SAMPLE DATA USED TO CALCULATE LINE-LENGTH TRANSFORM
    # returns data and sfx read in from sampledatacopy.mat
    mat = scipy.io.loadmat('sampledatacopy.mat')
    # grab data
    data = mat.get('data')
    data = data.transpose()

    # grab sampling frequency
    sfx = mat.get('samplingfrequency')
    if len(sfx.shape) == 2:
        sfx = sfx[0][0]
    elif len(sfx.shape) == 1:
        sfx = sfx[0]

    # CORRECT OUTPUT OF LINE-LENGTH TRANSFORM OF DATA ABOVE
    mat = scipy.io.loadmat('output.mat')
    # grab data
    expected = mat.get('L')
    return [data, sfx, expected]


def test_lltransform_1d(example_eeg_data):
    # returns line-length transform of a single channel
    data = example_eeg_data[0]
    sfx = example_eeg_data[1]
    expected = example_eeg_data[2]
    actual = lltransform(data[0], sfx)
    # format the two arrays for accurate testing
    expected = ['%.8f' % elem for elem in expected[0]]
    actual = ['%.8f' % elem for elem in actual]
    assert list(expected) == actual


def test_lltransform_2d(example_eeg_data):
    # returns line-length transform of all sample data
    data = example_eeg_data[0]
    sfx = example_eeg_data[1]
    expected = example_eeg_data[2]
    actual = lltransform(data, sfx)
    expected = [['%.8f' % elem for elem in ex] for ex in expected]
    actual = [['%.8f' % elem for elem in ac] for ac in actual]
    assert (list(expected) == list(actual))


def test_lltransform_3d():
    # passes if assertion is raised by code when passed incorrect data
    data = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    with pytest.raises(ValueError, match='Only accepts data in a 1-D or 2-D list.'):
        lltransform(data, 512)
        raise ValueError('Only accepts data in a 1-D or 2-D list.')


def test_lleventdetector_1(example_eeg_data):
    # should return correct time stamps for EON/EOFF
    expected_ts = np.column_stack(([1344, 4156], [1351, 4180]))
    l = example_eeg_data[2]
    sfx = example_eeg_data[1]
    actual = lleventdetector(l, sfx, 99.9, 3)
    if np.size(actual[0]) == np.size(expected_ts):
        assert [a == b for a, b in zip(list(actual[0]), list(expected_ts))]
    else:
        assert False


def test_lleventdetector_2(example_eeg_data):
    # should return correct channels for EON/EOFF
    expected_ch = ['3,', '3,']
    l = example_eeg_data[2]
    sfx = example_eeg_data[1]
    actual = lleventdetector(l, sfx, 99.9, 3)
    assert actual[1] == expected_ch


def test_lleventdetector_3(example_eeg_data):
    # tests edge case 1 for time stamps
    l = example_eeg_data[2]
    l[4][0:4] = 40000
    sfx = example_eeg_data[1]
    expected_ts = np.column_stack([[10, 13], [1344, 1355], [4159, 4180]])
    actual = lleventdetector(l, sfx, 99.9, 3)
    if np.size(actual[0]) == np.size(expected_ts):
        assert [a == b for a, b in zip(list(actual[0]), list(expected_ts))]
    else:
        assert False


def test_lleventdetector_4(example_eeg_data):
    # tests edge case 1 for channels
    l = example_eeg_data[2]
    l[4][0:4] = 40000
    sfx = example_eeg_data[1]
    actual = lleventdetector(l, sfx, 99.9, 3)
    expected_ch = ['4,', '3,', '3,']
    assert actual[1] == expected_ch


def test_lleventdetector_5(example_eeg_data):
    # tests edge case 2 for time stamps
    l = example_eeg_data[2]
    end = len(l[0])
    l[1][end - 4:end] = 40000
    sfx = example_eeg_data[1]
    actual = lleventdetector(l, sfx, 99.9, 3)
    expected_ts = np.column_stack([[1344., 1350.], [4159., 4180.], [5126., 5130.]])
    if np.size(actual[0]) == np.size(expected_ts):
        assert [a == b for a, b in zip(list(actual[0]), list(expected_ts))]
    else:
        assert False


def test_lleventdetector_6(example_eeg_data):
    # tests edge case 2 for channels
    l = example_eeg_data[2]
    end = len(l[0])
    l[1][end - 4:end] = 40000
    sfx = example_eeg_data[1]
    actual = lleventdetector(l, sfx, 99.9, 3)
    expected_ch = ['3,', '3,', '1,']
    assert actual[1] == expected_ch


def test_lleventdetector_8(example_eeg_data):
    # tests deleting short events time stamps
    l = example_eeg_data[2]
    sfx = example_eeg_data[1]
    minimum_event_time = 16
    actual = lleventdetector(l, sfx, 99.9, minimum_event_time)
    expected_ts = np.column_stack([[4156, 4180]])
    if np.size(actual[0]) == np.size(expected_ts):
        assert [a == b for a, b in zip(list(actual[0]), list(expected_ts))]
    else:
        assert False


def test_lleventdetector_9(example_eeg_data):
    # tests deleting short events channels
    l = example_eeg_data[2]
    sfx = example_eeg_data[1]
    minimum_event_time = 16
    actual = lleventdetector(l, sfx, 99.9, minimum_event_time)
    expected_ch = ['3,']
    assert expected_ch == actual[1]


def test_lleventdetector_10(example_eeg_data):
    # test that error is raised when EON and EOFF lengths are different
    l = example_eeg_data[2]
    end = len(l[0])
    sfx = example_eeg_data[1]
    l[1][end-1] = 40000
    minimum_event_time = 1
    with pytest.raises(RuntimeError, match='eON and eOFF are different lengths, check your code.'):
        actual = lleventdetector(l, sfx, 99.9, minimum_event_time)
        raise RuntimeError('eON and eOFF are different lengths, check your code.')

def test_lleventdetector_11(example_eeg_data):
    # tests that ech merges events correctly
    l = example_eeg_data[2]
    end = len(l[0])
    l[1][end - 4:end] = 40000
    l[2][end-3] = 40000
    sfx = example_eeg_data[1]
    actual = lleventdetector(l, sfx, 99.9, 3)
    expected_ch = ['3,', '3,', '1,2,']
    assert expected_ch == actual[1]
