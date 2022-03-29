import pandas as pd
from typing import List, Tuple
import numpy as np


df = pd.read_csv('houses_data.csv')

def read_csv_file(file_path: str) -> pd.DataFrame:  
    return pd.read_csv(file_path, usecols=cols_list)