"""
Models for runtime validation of the challenge app.
"""

# Standard Library
from typing import List
# Third-Party Libraries
from pydantic import BaseModel


# Core Classes
class FlighDetails(BaseModel):
    """
    Flight details model.

    Attributes
    ----------
    OPERA: str
        The airline operator.
    TIPOVUELO: str
        The flight type.
    MES: int
        The month of the flight.
    """
    OPERA: str
    TIPOVUELO: str
    MES: int


class PostRequest(BaseModel):
    """
    Post request model.

    Attributes
    ----------
    flights : List[Dict[str, Union[str, int]]]
        The list of flights to predict.
    """
    flights: List[FlighDetails]


class PostResponse(BaseModel):
    """
    Post response model.

    Attributes
    ----------
    predict : List[int]
        The list of predictions.
    """
    predict: List[int]
