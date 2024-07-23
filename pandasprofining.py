# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ydata_profiling import ProfileReport

# Assuming dataloader.py contains the code to load the dataframe
from dataloader import df

# Generate a profiling report
profile = ProfileReport(df, title="Profiling Report")

# Save the report to an HTML file
profile.to_file("pandasprofiling_output.html")

# Print confirmation
print("Profiling report has been generated and saved as pandasprofiling_output.html")
