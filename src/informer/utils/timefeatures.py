from typing import List

import numpy as np
try:
    import pandas as pd
    from pandas.tseries import offsets
    from pandas.tseries.frequencies import to_offset
except ModuleNotFoundError as error:
    if error.name == "pandas":
        raise ModuleNotFoundError(
            "Timestamp utilities require pandas; install the 'informer[time]' extra."
        ) from error
    raise

class TimeFeature:
    """Reference stateless calendar-feature interface (not an abstract class)."""
    def __init__(self):
        """Initialize a stateless feature."""
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Base placeholder returning None; subclasses encode DatetimeIndex values."""
        pass

    def __repr__(self):
        """Return the feature class name followed by parentheses."""
        return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    """Second of minute encoded as value between [-0.5, 0.5]."""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode valid seconds from index, dividing by 59 and centering."""
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode valid minutes from index, dividing by 59 and centering."""
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]."""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode valid hours from index, dividing by 23 and centering."""
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Weekday (Monday zero) encoded between [-0.5, 0.5]."""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode Monday-zero weekdays, dividing by six and centering."""
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode one-based month days with a fixed denominator of 30."""
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode one-based year days with a fixed denominator of 365."""
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode one-based months with a fixed denominator of 11."""
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode ISO weeks 1–53 with a fixed denominator of 52."""
        return (index.isocalendar().week.to_numpy(dtype=np.int64) - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        pandas offset string such as "12h", "5min", "1D", "ME", or "YE".

    Returns
    -------
    list[TimeFeature]
        Ordered feature instances; annual offsets return an empty list.
        Invalid strings raise pandas errors; unsupported offsets raise RuntimeError.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

def time_features(dates, timeenc=1, freq='h'):
    """
    Extract calendar columns from a DataFrame with a datetime-valued ``date`` column.

    Args:
        dates: DataFrame; categorical mode writes/overwrites calendar columns.
        timeenc: Zero for categories, one for continuous features; others return None.
        freq: Categorical key or pandas offset string, depending on timeenc.

    Returns:
        NumPy array (timestamps, features). Continuous mode does not mutate dates.
        Annual continuous selection raises because its feature list is empty.

    Categorical frequency keys (timeenc=0):
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    > 
    > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]): 
    > * QE - [month]
    > * ME - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * h - [Hour of day, day of week, day of month, day of year]
    > * min - [Minute of hour, hour of day, day of week, day of month, day of year]
    > * s - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    Categorical minute returns 0-3 for its 15 minute period; continuous minute
    uses raw minutes divided by 59, minus 0.5. Continuous keys follow pandas
    offsets, which differ from model embedding frequency keys.
    """
    if timeenc==0:
        dates['month'] = dates.date.apply(lambda row:row.month)
        dates['day'] = dates.date.apply(lambda row:row.day)
        dates['weekday'] = dates.date.apply(lambda row:row.weekday())
        dates['hour'] = dates.date.apply(lambda row:row.hour)
        dates['minute'] = dates.date.apply(lambda row:row.minute)
        dates['minute'] = dates.minute.map(lambda x:x//15)
        freq_map = {
            'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
            'b':['month','day','weekday'],'h':['month','day','weekday','hour'],
            't':['month','day','weekday','hour','minute'],
        }
        return dates[freq_map[freq.lower()]].values
    if timeenc==1:
        dates = pd.to_datetime(dates.date.values)
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1,0)
