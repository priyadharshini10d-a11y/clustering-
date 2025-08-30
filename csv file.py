import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    'CustomerID': range(1, 201),
    'Age': np.random.randint(18, 70, 200),
    'Annual Income (k$)': np.random.randint(15, 150, 200),
    'Spending Score (1-100)': np.random.randint(1, 100, 200)
}

df = pd.DataFrame(data)
df.to_csv('customers.csv', index=False)
print("customers.csv with 200 rows created successfully!")
