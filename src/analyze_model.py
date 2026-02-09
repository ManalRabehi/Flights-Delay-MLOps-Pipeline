import pandas as pd 

df = pd.read_csv("data/processed/processed_data.csv")

y = (df['arr_delay'] > 0).astype(int)

print("Répartition des classes : ")
print(y.value_counts())
print("\nProportion des classes : ")
print(y.value_counts(normalize=True))