"""Calendar ordering, ranges, mutation, and modern pandas compatibility."""

import numpy as np
import pytest

pd = pytest.importorskip('pandas')
from informer.utils.timefeatures import time_features, time_features_from_frequency_str, WeekOfYear


@pytest.mark.parametrize('freq,names', [
    ('YE', []), ('QE', ['MonthOfYear']), ('ME', ['MonthOfYear']),
    ('W', ['DayOfMonth', 'WeekOfYear']),
    ('D', ['DayOfWeek', 'DayOfMonth', 'DayOfYear']),
    ('B', ['DayOfWeek', 'DayOfMonth', 'DayOfYear']),
    ('h', ['HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear']),
    ('15min', ['MinuteOfHour', 'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear']),
    ('s', ['SecondOfMinute', 'MinuteOfHour', 'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear']),
])
def test_frequency_order(freq, names):
    """Keep the exact channel order for supported pandas offset families."""
    assert [type(item).__name__ for item in time_features_from_frequency_str(freq)] == names


def test_iso_week_year_boundary():
    """Recover historical ISO weeks across week 53 and a calendar-year boundary."""
    index = pd.to_datetime(['2020-12-31', '2021-01-01', '2021-01-04'])
    np.testing.assert_allclose(WeekOfYear()(index), [0.5, 0.5, -0.5])


def test_continuous_values_and_no_mutation():
    """Use known leap-year end points and check all second-level columns."""
    frame = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-12-31 23:59:59'], format='mixed')})
    original = frame.copy(deep=True)
    result = time_features(frame, freq='s')
    np.testing.assert_allclose(result, [[-.5, -.5, -.5, 2 / 6 - .5, -.5, -.5],
                                       [.5, .5, .5, 3 / 6 - .5, .5, .5]])
    pd.testing.assert_frame_equal(frame, original)
    assert np.all((result >= -.5) & (result <= .5))


@pytest.mark.parametrize('freq,width', [('y', 0), ('m', 1), ('w', 1), ('d', 3), ('b', 3), ('h', 4), ('t', 5)])
def test_categorical_mutation_and_minute_bins(freq, width):
    """Retain frame mutation and 15-minute categories even for coarser outputs."""
    frame = pd.DataFrame({'date': pd.to_datetime(['2024-02-29 12:00', '2024-02-29 12:14',
                                                '2024-02-29 12:15', '2024-02-29 12:30',
                                                '2024-02-29 12:59']), 'minute': -1})
    result = time_features(frame, timeenc=0, freq=freq)
    assert result.shape == (5, width)
    np.testing.assert_array_equal(frame.minute, [0, 0, 1, 2, 3])
    if freq == 't':
        np.testing.assert_array_equal(result[2], [2, 29, 3, 12, 1])


def test_unsupported_and_annual_cases():
    """Preserve explicit reference limitations without adding fallback features."""
    frame = pd.DataFrame({'date': pd.to_datetime(['2024-01-01'])})
    with pytest.raises(ValueError):
        time_features(frame, freq='YE')
    with pytest.raises(RuntimeError):
        time_features_from_frequency_str('ms')
    with pytest.raises(KeyError):
        time_features(frame, timeenc=0, freq='s')
    assert time_features(frame, timeenc=99) is None
