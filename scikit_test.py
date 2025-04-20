import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import onnxruntime as rt

df = pd.read_csv('titanic.csv') # Parse titanic.csv as pandas DataFrame.

# Get X (feature variables) and y (target variable) from the pandas DataFrame:
X = df.drop("Survived", axis=1)
y = df["Survived"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) # Split X and y in train and test panda DataFrame. 0 is assigned to random_state in order to get the same split always (for learning purposes).

X_train_fill_na = X_train.copy() # X_train_fill_na saves a X_train copy.
X_train_fill_na.fillna({"Cabin":"cabin_missing", "Embarked":"embarked_missing"}, inplace=True) # N/A cells are filled with respective attribute fillers. 

imputer = SimpleImputer() # SimpleImputer object is declared.

# imputer is configured:
imputer.fit(X_train_fill_na[["Age"]])
age_imputed = pd.DataFrame( # age column is transformed:
    imputer.transform(X_train_fill_na[["Age"]]),
    index=X_train_fill_na.index,
    columns=["Age"]
)

X_train_fill_na["Age"] = age_imputed #X_train_fill_na Age column is assigned to age_imputed.

categorical_features = ["Sex", "Cabin", "Embarked"] # Columns selected for dummying.
X_train_categorical = X_train_fill_na[categorical_features.copy()] # Selected columns are saved in X_train_categorical. Null values are filled.

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False) # OneHotEncoder object is declared.

# ohe is configured:
ohe.fit(X_train_categorical) 
X_train_ohe = pd.DataFrame( # X_train_categorical columns are transformed:
    ohe.transform(X_train_categorical), 
    index=X_train_categorical.index,
    columns=np.hstack(ohe.categories_)
)

numeric_features = ["Pclass", "Age", "SibSp", "Fare"] # Columns selected for normalization.

X_train_numeric = X_train_fill_na[numeric_features].copy() # Selected columns are saved in X_train_numeric. Null values are filled.

scaler = MinMaxScaler() # MinMaxScaler object is declared.

# scaler is configured:
scaler.fit(X_train_numeric)
X_train_scaled = pd.DataFrame(
    scaler.transform(X_train_numeric), # X_train_numeric columns are transformed:
    index=X_train_numeric.index,
    columns=X_train_numeric.columns
)

X_train_full = pd.concat([X_train_scaled, X_train_ohe], axis=1) # Every X_train subset is concatenated.

logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear') # LogisticRegression object is declared.

model_log = logreg.fit(X_train_full, y_train) # Logistic regression model fits with X_train_full and y_train.

y_hat_train = logreg.predict(X_train_full) # Model predicts X_train_full, and saves the results in y_hat_train.

train_residuals = np.abs(y_train - y_hat_train) # y_train minus y_hat_train gives a vector. In this vector, rows with values different from 0 are y_hat_train prediction misses, while rows with value equal to 0 are y_hat_train prediction hits. 
print(pd.Series(train_residuals, name="Residuals (counts)").value_counts()) # Prints absolute prediction hits and misses.
print() # Prints a breakline.
print(pd.Series(train_residuals, name="Residuals (proportions)").value_counts(normalize=True)) # Prints relative prediction hits and misses.

# Filling in missing categorical data:
X_test_fill_na = X_test.copy()
X_test_fill_na.fillna({"Cabin":"cabin_missing", "Embarked":"embarked_missing"}, inplace=True)

# Filling in missing numeric data:
test_age_imputed = pd.DataFrame(
    imputer.transform(X_test_fill_na[["Age"]]),
    index=X_test_fill_na.index,
    columns=["Age"]
)
X_test_fill_na["Age"] = test_age_imputed

# Handling categorical data:
X_test_categorical = X_test_fill_na[categorical_features].copy()
X_test_ohe = pd.DataFrame(
    ohe.transform(X_test_categorical),
    index=X_test_categorical.index,
    columns=np.hstack(ohe.categories_)
)

# Normalization:
X_test_numeric = X_test_fill_na[numeric_features].copy()
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_numeric),
    index=X_test_numeric.index,
    columns=X_test_numeric.columns
)

# Concatenating categorical and numeric data:
X_test_full = pd.concat([X_test_scaled, X_test_ohe], axis=1)

y_hat_test = logreg.predict(X_test_full) # Model predicts X_test_full, and saves the results in y_hat_test.

test_residuals = np.abs(y_test - y_hat_test) # y_test minus y_hat_test gives a vector. In this vector, rows with values different from 0 are y_hat_train prediction misses, while rows with value equal to 0 are y_hat_train prediction hits. 
print(pd.Series(test_residuals, name="Residuals (counts)").value_counts()) # Prints absolute prediction hits and misses.
print() # Prints a breakline.
print(pd.Series(test_residuals, name="Residuals (proportions)").value_counts(normalize=True)) # Prints absolute prediction hits and misses.

X_test_full = X_test_full.to_numpy() # Pandas Dataframe is casted to NumPy ndarray

initial_type = [('float_input', FloatTensorType([None, X_test_full[0].size]))] # Each X_test_full row has the same number of columns than X_test_full[0]. That's X_test_full[0].size.
onnx = convert_sklearn(logreg, initial_types=initial_type) # onnx object is declared with logreg classifier and initial_type.
with open("logreg_titanic.onnx", "wb") as f: # onnx file is created:
    f.write(onnx.SerializeToString()) # onnx object is written in file.

sess = rt.InferenceSession("logreg_titanic.onnx", providers=["CPUExecutionProvider"]) # sess object is declared to run onnx file model in CPU.
# Input and output names are saved:
input_name = sess.get_inputs()[0].name 
label_name = sess.get_outputs()[0].name

pred_onnx = sess.run([label_name], {input_name: X_test_full.astype(np.float32)})[0] # sess predicts X_test_full (as type float32) and saves results in pred_onnx.
# y_hat_test (scikit-learn prediction) and pred_onnx (ONNX prediction) are printed. Numbers are the same. Thus, ONNX works!
print("y_hat_test:")
print(y_hat_test)
print("pred_onnx:")
print(pred_onnx)