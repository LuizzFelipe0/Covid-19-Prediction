import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import uvicorn


# Pydantic model for the request data
class InputData(BaseModel):
    Tosse_Seca: float = Field(..., example=3)
    Febre_Alta: float = Field(..., example=1)
    Dor_de_Garganta: float = Field(..., example=0)
    Dificuldade_em_Respirar: float = Field(..., example=5)


# Pydantic model for the response data
class OutputData(BaseModel):
    Prediction: float
    Result: str

# Create a FastAPI instance
app = FastAPI(title='COVID-19 Prediction')

# Load the dataset
data = pd.read_excel('./covid19-symptoms-dataset.xlsx')

def train_model():
    # Separate the features (X) and labels (y)
    X = data.drop('Infected with Covid19', axis=1)
    y = data['Infected with Covid19']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a logistic regression model
    model = LogisticRegression()
    # Train the model on the training data
    model.fit(X_train, y_train)
    return model


# Train the model and store it in a variable
model = train_model()


# Make predictions using the trained model
def predict(model, input_data):
    probabilities = model.predict_proba(input_data)
    return probabilities[:, 1]  # Return the probabilities of positive class



@app.get("/root", tags=["Raiz"])
def root():
    return {"message": "Simple ML API to predict Covid-19 diagnosys"}


# Route for making predictions
@app.post("/predict", response_model=List[OutputData], status_code=200 ,tags=["Previsão"])
def make_predictions(data: List[InputData]):
    # Convert the input data to DataFrame
    input_data = pd.DataFrame([item.dict() for item in data])

    # Make predictions on the input data
    probabilities = predict(model, input_data)

    results = []
    for probability in probabilities:
        percentage_chance = round(probability * 100, 2)
        if percentage_chance > 69:
            result = "Alta Chance de possuir Covid-19"
        elif 40 < percentage_chance <= 69:
            result = "Média Chance de possuir Covid-19"
        else:
            result = "Pequena chance de possuir Covid-19"

        results.append({"Previsão (%)": percentage_chance, "Resultado Aproximado": result})

    return results



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
