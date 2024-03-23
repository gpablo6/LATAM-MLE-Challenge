"""
Constant values to be utiltized by the model and the utilities functions.
"""

# Modify if deem necessary without changing the model module.
DELAY_TRESHOLD = 15
TOP_FEATURES = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
]
HYPERPARAMETERS = {
    'test_size': 0.33,
    'random_state': 1,
    'learning_rate': 0.01,
    # Set to True if you want to use the calculation.
    'scale_pos_weight': True  # For more information go to `utils` module.
}
