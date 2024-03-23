"""
Utilities functions for the model.
"""

# Standard library imports
from datetime import datetime
# Third-party imports
import numpy as np
import pandas as pd


# Functions
def get_min_diff(data: pd.Series) -> pd.Series:
    """
    Get the difference in minutes between two dates.

    Parameters
    ----------
    data: pd.Series
        Row with the values for columns 'Fecha-O' and 'Fecha-I'.

    Returns
    -------
    pd.Series
        Series with the difference in minutes.
    """
    # Convert to datetime
    fecha_o = datetime.strptime(
        data['Fecha-O'],
        '%Y-%m-%d %H:%M:%S'
    )
    fecha_i = datetime.strptime(
        data['Fecha-I'],
        '%Y-%m-%d %H:%M:%S'
    )
    # Calculate the difference in minutes
    min_diff = (
        (
            fecha_o -
            fecha_i
        ).total_seconds()
    ) / 60
    # Return the result
    return min_diff


def calculate_delay(
    data: pd.DataFrame,
    treshold: int = 15
) -> pd.Series:
    """
    Calculate the delay flag given a time column in minutes and a treshold.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame with the column `min_diff`.
    treshold: int, default = 15
        Treshold in minutes to consider a delay.

    Returns
    -------
    pd.Series
        Series with the delay in minutes.
    """
    # Check if the column is in the DataFrame
    if 'min_diff' not in data.columns:
        raise ValueError(
            'Column `min_diff` not found in the DataFrame.'
        )
    # Calculate the delay
    return np.where(
        data['min_diff'] > treshold,
        1,
        0
    )


def get_balance_scale(
    target: pd.Series
) -> dict:
    """
    Get the balance scale for the target variable.

    Parameters
    ----------
    target: pd.Series
        Series with the target variable.

    Returns
    -------
    dict
        Dictionary with the balance scale.
    """
    # Get the balance scale
    balance = target[target == 0].size / target[target == 1].size
    # Return the result
    return {
        'scale_pos_weight': balance
    }
