import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataframe
from dataloader import df

# Display a sample of 5 rows
print("Sample of 5 rows:")
print(df.sample(5))

# Display the shape of the dataframe
print("\nShape of the dataframe:")
print(df.shape)

# Display unique values in the 'Class' column
print("\nUnique values in 'Class' column:")
print(df["Class"].unique())

# Display descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Display the count of missing values in each column
print("\nCount of missing values in each column:")
print(df.isnull().sum())

# Display information about the dataframe
print("\nDataframe info:")
print(df.info())

# Display value counts of the 'Class' column
print("\nValue counts of 'Class' column:")
class_counts = df["Class"].value_counts()
print(class_counts)

# Save all values to a file
with open('datadescription.txt', 'w') as f:
    f.write("Sample of 5 rows:\n")
    f.write(df.sample(5).to_string())
    f.write("\n\nShape of the dataframe:\n")
    f.write(str(df.shape))
    f.write("\n\nUnique values in 'Class' column:\n")
    f.write(np.array2string(df["Class"].unique()))
    f.write("\n\nDescriptive statistics:\n")
    f.write(df.describe().to_string())
    f.write("\n\nCount of missing values in each column:\n")
    f.write(df.isnull().sum().to_string())
    f.write("\n\nDataframe info:\n")
    df.info(buf=f)
    f.write("\n\nValue counts of 'Class' column:\n")
    f.write(class_counts.to_string())

print("\nAll values have been printed to 'dataframe_values.txt'")
