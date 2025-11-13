from flask import Flask
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("house_price.csv")

X = df.iloc[:,0:2]
y = df["price"]


X_train, X_test, y_trian, y_test = train_test_split(X,y,train_size=0.25,random_state=0)

app = Flask(__name__)


with open("lrmodel.pkl","rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return "Model is Running"

@app.route("/predict",methods=["GET"])
def predict():
    pred = model.predict([[3,1930]])
    y_pred = model.predict(X_test)
    # acc = accuracy_score(y_test,y_pred)
    return f"House Price Prediction for 3 BHK and 1950 sqft_living: {pred}"
    

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)