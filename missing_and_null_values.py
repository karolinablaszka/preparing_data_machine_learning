import pandas as pd
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor



file_path = "C:\\Users\\karol\\machine_learning\\"

df = pd.read_csv('houses_data.csv')
col_list = list(df)

#wyświetl liczbę (tablica + bar plot z %) brakujących wartości per cecha (kolumna)
def null_values():
    df.isna().sum()[df.isna().sum()>0]
    plt.title("Percentage of NaN in every column")

    df_percent = null_values.div(len(df), axis=0).round(4)*100

    #ax = df_percent.plot(kind='bar', figsize=(6,6), width=0.5)
    plt.xticks(rotation = 0)
    plt.show() 



#sprawdź typy danych
data_types = df.dtypes
print(f"Data types: {data_types}")

#sprawdź czy są duplikaty (opcjonalnie)
data_if_duplicated = df.duplicated(keep = "last").iloc[0] 
print(f"Data is duplicated: {data_if_duplicated}")

#wyznacz korelacje cech vs kolumna 'Price'
col_correlation = df.corr()
print(f"Correlation: {col_correlation}")

#Zadanie główne: 1. Wczytaj ponownie plik (utwórz funkcję), gdzie:
#będą pobierane tylko wybrane kolumny,
#zostaną zdefiniowane typy ww. kolumn.

def read_file(df):
    df_2 = df.select_dtypes(include=["int16", "int32", "int64", "float16", "float32", "float64"])
    return df_2

#2. Przygotuj kilka metod/strategii do zastępowania brakujących wartości:
#wybierz właściwe kolumny
#zastanów się czy każda metoda jest właściwa dla całego datasetu
#wybierz tylko kolumny z wartościami numerycznymi,
#zastosuj wybrane metody i porównaj metrykę
#zmień LinearRegression na RandomForestRegressor (opcjonalnie)
#sprawdź jakie będzie MAE bez uzupełniania brakujących wartości


df_2 = df.select_dtypes(include=["int16", "int32", "int64", "float16", "float32", "float64"])
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None)
mean_imputer_transform = mean_imputer.fit_transform(df_2[['YearBuilt', 'Car']])
print (f'SimpleImputer: {mean_imputer_transform}')
    
knn = KNNImputer(missing_values=np.nan)
knn_transform = knn.fit_transform(df_2)
print (f'KNNImputer: {knn_transform}')

houses_predictors = df[['Rooms', 'Bedroom2']]
houses_target = df['Price']

X_train, X_test, y_train, y_test = train_test_split(houses_predictors, houses_target, train_size=0.7, test_size=0.3, random_state=42
)

def score_dataset(X_train, X_test, y_train, y_test):
    regr_model = RandomForestRegressor()
    regr_model.fit(X_train, y_train)
    preds = regr_model.predict(X_test)
    return mean_absolute_error(y_test, preds)


MAE = score_dataset(X_train, X_test, y_train, y_test)
print(f'MAE: {MAE}')

