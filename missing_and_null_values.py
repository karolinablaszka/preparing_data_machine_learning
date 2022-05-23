import pandas as pd
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

file_path = "C:\\Users\\karol\\machine_learning\\"

df = pd.read_csv('houses_data.csv')
col_list = list(df)

#wyświetl liczbę (tablica + bar plot z %) brakujących wartości per cecha (kolumna)
null_values = df.isna().sum()[df.isna().sum()>0]

plt.title("Percentage of NaN in every column")

df_percent = null_values.div(len(df), axis=0).round(4)*100

ax = df_percent.plot(kind='bar', figsize=(6,6), width=0.5)
plt.xticks(rotation = 0)
plt.show() 




# def main(file_path: str) -> pd.DataFrame:  
#     return pd.read_csv(file_path, usecols=col_list)

# if __name__=='__main__':
#     main()