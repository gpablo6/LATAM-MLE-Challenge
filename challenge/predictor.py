"""
Predictor service for model.
"""

# Third-Party Libraries
import pandas as pd
from fastapi import HTTPException
# Local Package
from .model import DelayModel
from .utils import get_root_path
from .dataclass import PostRequest, PostResponse


# Core Class
class Predictor(object):
    """
    Predictor service for model.

    Attributes
    ----------
    _model_instance: DelayModel
        The singleton instance of the model.
    _data: pd.DataFrame
        The data available for model inference.
    """

    _model_instance: DelayModel

    # Dunder methods
    def __new__(cls):
        """
        Dunder method for initializing and returning the singleton instance.

        Returns
        -------
        DelayModel
            The singleton instance.
        """
        if not hasattr(cls, '_instance'):
            cls._instance = super(
                Predictor,
                cls
            ).__new__(
                cls
            )
            # Initialize the object
            cls._instance._init_logic()
        return cls._instance

    def _init_logic(self) -> None:
        """
        Initialize the logic for the predictor.
        """
        # Create model instance
        self._model_instance = DelayModel()
        # Load the model
        self._model_instance._load_model()
        # Load the data
        root_path = get_root_path()
        data_path = root_path / 'data' / 'data.csv'
        self._data = pd.read_csv(data_path)

    def predict(
        self,
        payload: PostRequest
    ) -> PostResponse:
        """
        Predict the delay for the given data.

        Args:
            data (pd.DataFrame): data to predict.

        Returns:
            pd.DataFrame: predictions.
        """
        predictions = []
        for flight in payload.flights:
            try:
                # NOTE: This process could be optimized if not dependant over
                # a single file by having it indexed and previously store ready
                # for inference.
                row = self._data[
                    (self._data['OPERA'] == flight.OPERA) &
                    (self._data['TIPOVUELO'] == flight.TIPOVUELO) &
                    (self._data['MES'] == flight.MES)
                ]
                # Preprocess the data
                if not row.empty:
                    features = self._model_instance.preprocess(
                        pd.DataFrame(payload.flights)
                    )
                    # Predict the delay
                    result = self._model_instance.predict(features)
                    # Add the prediction to the final result
                    predictions.append(
                        result[0]
                    )
                else:
                    raise ValueError(
                        f"Flight not found:\t {flight.dict()}"
                    )
            except Exception as e:
                # Log the issue.
                print(f"[ERROR]: {e}")
                # Handle exceptions for avoiding stopping the service.
                raise HTTPException(
                    status_code=400,
                    detail='Invalid flight'
                )
        # Return the response
        return PostResponse(
            predict=predictions
        )
