from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
import pandas as pd
from typing import List
import os

# Import your DelayModel class and any necessary preprocessing functions
from model import DelayModel

app = FastAPI()

model = DelayModel() 
model.load_model(os.environ.get('FLIGHT_DELAY_MODEL'))


class FlightInput(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class BatchInput(BaseModel):
    flights: List[FlightInput]
    

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict")
async def predict(data: BatchInput):
    try:
        # Extract the list of flight data from the request
        flight_data = data.flights

        # Convert input data to a DataFrame
        df = pd.DataFrame([flight.dict() for flight in flight_data])

        # Preprocess the data using your DelayModel's preprocess method
        features = model.preprocess(df)

        # Make predictions using your DelayModel's predict method
        predictions = model.predict(features)

        return {"predict": predictions}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
