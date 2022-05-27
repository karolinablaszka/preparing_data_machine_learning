import pandas as pd
from typing import List, Dict, Callable
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore as zscore_outlier, median_abs_deviation

file_path = "C:\\Users\\karol\\machine_learning\\"

df = pd.read_csv('houses_data.csv')
col_list = list(df)

#Zadanie główne: Napisz funkcję, 
#które będą usuwały wartości odstające przy wykorzystaniu:
# -log transform ??
# -removing 0.1 & 0.9 percentile
# -IQR
#  df < (Q1 - 1.5 * IQR))|(df > (Q3 + 1.5 * IQR)
# -z-score (2 i/lub 3 SD)
# -modified Z-score
# Porównaj wyniki przez: -policzenie liczby wystąpień wartości odstających,
# -wyznaczenie MAE (kod z poprzednich zajęć)

df = df.select_dtypes(include=["int16", "int32", "int64", "float16", "float32", "float64"])

#def percentiles_outliers(df):
    #lower_percentile = 0.1
    #higher_percentile = 0.9
    #low, high = df.quantile([lower_percentile, higher_percentile])
    #return (df[df < low]) & (df[df > high])
        

def iqr_outliers(df):
    Q1 = df.quantile(0.1)
    Q3 = df.quantile(0.9)
    IQR = Q3 - Q1
    #return (df < (Q1 - 1.5 * IQR))&(df > (Q3 + 1.5 * IQR))
    print (f'dwa: {(df < (Q1 - 1.5 * IQR))&(df > (Q3 + 1.5 * IQR))}')


def z_score(df):
    outlier = []
    for i in df:
        z = (i-np.mean(df))/np.std(df)
        if z > 3:
            outlier.append(i)
    #return outlier
    print (f'trzy: {outlier}')


def modified_z_score_outlier(df):    
    mad_column = median_abs_deviation(df)
    median = np.median(df)
    mad_score = np.abs(0.6745 * (df - median) / mad_column)
    #return mad_score > 3.5
    print (f'cztery: {mad_score > 3.5}')    


# df_2 = df.select_dtypes(include=["int16", "int32", "int64", "float16", "float32", "float64"])
# mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None)
# mean_imputer_transform = mean_imputer.fit_transform(df_2[['YearBuilt', 'Car']])
# print (f'SimpleImputer: {mean_imputer_transform}')
    
knn = KNNImputer(missing_values=np.nan)
knn_transform = knn.fit_transform(df)
print (f'KNNImputer: {knn_transform}')


 
def score_dataset(X_train, X_test, y_train, y_test):
     regr_model = LinearRegression()
     regr_model.fit(X_train, y_train)
     preds = regr_model.predict(X_test)
     return mean_absolute_error(y_test, preds)


#MAE = score_dataset(X_train, X_test, y_train, y_test)
#print(f'MAE: {MAE}')

def outliers(df_copy: pd.DataFrame, outliers_methods_dict: Dict[str, Callable]):

    for method_name, method in outliers_methods_dict.items():

        print("\nRunning method: ", method_name)

        df = df_copy.copy()

        print('\nOutliers:\n\n', df.apply(lambda x: method(x)).sum(), '\n\n')

        df = df[df.apply(lambda x: ~method(x))].dropna()

        houses_predictors = df[['Rooms', 'Bedroom2']]
        houses_target = df['Price']

        X_train, X_test, y_train, y_test = train_test_split(houses_predictors, houses_target, train_size=0.7, test_size=0.3, random_state=0
        )

        print("\nMAE: ", score_dataset(X_train, X_test, y_train, y_test), "\n\n\n\n")


def outliers_results():
    outliers_methods_dict = {
        "percentiles_outliers": percentiles_outliers,
        "z_score": z_score,
        "mod_z_score": modified_z_score_outlier,
        "iqr_outliers": iqr_outliers
    }
    outliers(df, outliers_methods_dict)



if __name__ == '__main__':
    outliers_results()