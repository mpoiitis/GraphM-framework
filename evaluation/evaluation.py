import pandas as pd

def evaluation(input):
    df = pd.read_csv(input)
    print(df.head(5))