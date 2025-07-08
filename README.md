# Pima Indians Diabetes Prediction using ANN

This project builds an Artificial Neural Network (ANN) model to predict whether a patient has diabetes, based on medical diagnostic measurements from the **Pima Indians Diabetes Dataset**.

---

## 📊 Dataset Description

The **Pima Indians Diabetes Dataset** is provided by the UCI Machine Learning Repository. It contains medical data for female patients of Pima Indian heritage who are at least 21 years old.

- **Number of instances:** 768  
- **Number of features:** 8 input features + 1 binary output  
- **Prediction Task:** Binary classification — whether the patient has diabetes (1) or not (0)

---

## 🧬 Feature Details

| Feature Name                 | Description                                                |
|-----------------------------|------------------------------------------------------------|
| Pregnancies                 | Number of times pregnant                                   |
| Glucose                     | Plasma glucose concentration after 2 hours in OGTT         |
| BloodPressure               | Diastolic blood pressure (mm Hg)                          |
| SkinThickness               | Triceps skinfold thickness (mm)                            |
| Insulin                     | 2-Hour serum insulin (mu U/ml)                             |
| BMI                         | Body mass index (weight in kg/(height in m)^2)             |
| DiabetesPedigreeFunction    | Diabetes pedigree function (genetic risk)                 |
| Age                         | Age of the patient (in years)                              |
| Outcome                     | Class variable (0 = non-diabetic, 1 = diabetic)            |

---

## 🧠 Model Architecture

I implemented a simple Artificial Neural Network using Keras with the following architecture:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(32, activation='relu',  input_shape = (8, )))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## 🔧 Hyperparameter Tuning

I performed hyperparameter tuning using **Keras Tuner** to search for the best combination of:

- Number of units in each layer  
- Activation functions  
- Optimizer choice

The tuning was done using the `RandomSearch` strategy from `keras_tuner`.

✅ **After tuning, the model achieved an accuracy of approximately 75% on the validation set.**
