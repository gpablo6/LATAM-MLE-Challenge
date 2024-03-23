"""
Entrypoint for the FastAPI application.
"""

# Third-party libraries
import fastapi
# Local Package
from .dataclass import PostRequest
from .predictor import Predictor

app = fastapi.FastAPI()


# Lifespan Events
@app.on_event("startup")
async def startup_event():
    """
    Event triggered when the service starts.

    Notes
    -----
    The Predictor is initialized to have the model already loaded in memory.
    """
    # Initialize the predictor instance once.
    _ = Predictor()


# Endpoints
@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Health check endpoint.

    Returns
    -------
    dict
        The status of the service.
    """
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(prediction_request: PostRequest) -> dict:
    """
    Endpoint for predicting the delay of flights.

    Parameters
    ----------
    prediction_request : PostRequest
        The request containing the data to predict.

    Returns
    -------
    dict
        The prediction for the given data.
    """
    # Get the predictor instance
    predictor = Predictor()
    # Predict the delay
    prediction = predictor.predict(
        prediction_request
    )
    # Return the prediction
    return prediction.dict()
