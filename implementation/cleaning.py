import pandas as pd
import numpy as np

data = pd.read_csv("cardio_train_dataset.csv", delimiter=";")

data["age"] = (data["age"] / 365.25).round(2)
data["gender"] = data["gender"].map({1: 0, 2: 1})

# Feature engineering (BMI)
data["bmi"] = data["weight"] / (data["height"] / 100) ** 2

ordinal_cols = ["cholesterol", "gluc"]
data[ordinal_cols] = data[ordinal_cols].astype(int)


data = data[(data["height"] >= 130) & (data["height"] <= 210)]
data = data[(data["weight"] >= 30) & (data["weight"] <= 210)]
data = data[(data["ap_hi"] >= 70) & (data["ap_hi"] <= 210)]
data = data[(data["ap_lo"] >= 50) & (data["ap_lo"] <= 150)]

mask = data["ap_lo"] > data["ap_hi"]
data.loc[mask, ["ap_hi", "ap_lo"]] = data.loc[mask, ["ap_lo", "ap_hi"]].values

data = data.drop_duplicates()
data = data.dropna()

# Feature engineering (pulse pressure)
data["pulse_pressure"] = data["ap_hi"] - data["ap_lo"]

binary_cols = ["smoke", "alco", "active"]
data[binary_cols] = data[binary_cols].astype(bool)


#sklearn for future normalizaiton:

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# numerical_cols = ["age", "height", "weight", "ap_hi", "ap_lo", "pulse_pressure"]
# data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

data.to_csv("cleaned_cardio_train_dataset.csv", index=False)
