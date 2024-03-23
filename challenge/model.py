"""
Implementation of the core class for the model.
"""

# Third-Party Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
# Standard Library
from typing import Tuple, Union, List
from os import path
import pickle
from pathlib import Path
# Local Package
from .utils import get_min_diff, calculate_delay, get_balance_scale
from .constants import TOP_FEATURES, DELAY_TRESHOLD, HYPERPARAMETERS


# Core Class
class DelayModel:
    """
    Core class for model workflow.
    """

    def __init__(
        self
    ):
        """
        Initialize the model class.
        """
        self._model = None  # Model should be saved in this attribute.
        # Define the path to save the model.
        self._model_path = path.join(
            Path(__file__).parent.absolute(),
            'model.pkl'
        )

    def _load_model(self) -> None:
        """
        Load the latest available model.
        """
        if self._model is None:
            with open(self._model_path, 'rb') as file:
                self._model = pickle.load(file)

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Check if the columns match the expected ones.
        target_cols = [
            'OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM',
            'Fecha-O', 'Fecha-I'
        ]
        if not set(target_cols).issubset(data.columns):
            raise ValueError(
                'Columns do not match the expected ones.'
            )
        # Calculate the label.
        data['min_diff'] = data.apply(
            get_min_diff,
            axis=1
        )
        data['delay'] = calculate_delay(
            data,
            treshold=DELAY_TRESHOLD
        )
        # Generate the features and target datasets.
        features = pd.concat(
            [
                pd.get_dummies(
                    data['OPERA'], prefix='OPERA'
                ),
                pd.get_dummies(
                    data['TIPOVUELO'], prefix='TIPOVUELO'
                ),
                pd.get_dummies(
                    data['MES'], prefix='MES'
                )
            ],
            axis=1
        )
        target = data[['delay']]
        # Filter features based on importance analysis.
        if TOP_FEATURES:
            features = features[TOP_FEATURES]
        # Return the data.
        if target_column:
            return (
                features,
                target,
            )
        else:
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Split the data into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            features,
            target,
            test_size=HYPERPARAMETERS['test_size'],
            random_state=42
        )
        # Define the model.
        if HYPERPARAMETERS['scale_pos_weight']:
            balance_scale = get_balance_scale(target)
            self._model = xgb.XGBClassifier(
                random_state=HYPERPARAMETERS['random_state'],
                learning_rate=HYPERPARAMETERS['learning_rate'],
                scale_pos_weight=balance_scale['scale_pos_weight']
            )
        else:
            self._model = xgb.XGBClassifier(
                random_state=HYPERPARAMETERS['random_state'],
                learning_rate=HYPERPARAMETERS['learning_rate']
            )
        # Fit the model.
        self._model.fit(
            X_train,
            y_train
        )
        # Evaluate the model.
        y_pred = self._model.predict(X_val)
        print(
            classification_report(y_val, y_pred)
        )
        # Save the model.
        # This could be updated to keep registry of the trained models.
        with open(self._model_path, 'wb') as file:
            pickle.dump(self._model, file)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        # Check the model is correctly loaded.
        if self._model is None:
            raise ValueError(
                'Model not loaded.'
            )
        # Predict the target.
        return self._model.predict(features)
