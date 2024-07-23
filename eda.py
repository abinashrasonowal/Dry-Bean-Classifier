import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataframe
from dataloader import df

indicators = df.columns[:-1]
ax = df.plot(x="Class", y=indicators, subplots=True, layout=(4, 4),figsize=(16, 16), sharex=False, rot=90)

# # Save the plot
# # plt.show(
# plt.savefig('./eda_outputs/indicators_plot.png')
# print('indicators plot saved.....')

# unique_classes = df.Class.unique()
# def draw_conditional_distribution(ax, df, col):
#     bins = np.linspace(df[col].min(), df[col].max(), 50)
#     for cls in unique_classes:
#         ax.hist(df[df.Class == cls][col], alpha=0.5, label=cls, bins=bins)
#     ax.set_title(f'Distributions for {col}')
#     ax.legend()

# fig, ax = plt.subplots(len(df.drop('Class', axis=1).columns) // 4, 4, figsize=(15, 15))
# for idx, col in enumerate(df.drop('Class', axis=1).columns):
#     draw_conditional_distribution(ax[idx // 4, idx % 4], df, col)
# plt.suptitle('Distributions for all features, conditional on target')
# plt.tight_layout()
# plt.xticks(rotation = 90)
# plt.savefig('./eda_outputs/conditional_distribution.png')
# print('conditional distribution saved....')

# plt.figure(figsize = (10, 5))
# sns.boxplot(df[indicators])
# plt.title("Boxplot of Dry Bean")
# plt.xticks(rotation = 90)
# plt.savefig('./eda_outputs/boxplot.png')
# print('boxplot saved....')


# # removed = df[(df["Area"] >= 100000) & (df["ConvexArea"] >= 100000)]

# df = df[(df["Area"] < 100000) | (df["ConvexArea"] < 100000)]

# plt.figure(figsize = (10, 5))
# sns.boxplot(df[indicators])
# plt.title("Boxplot of Dry Bean after removal of outliers")
# plt.xticks(rotation = 90)
# plt.savefig('./eda_outputs/boxplot_after_removal.png')
# print('boxplot after removal saved....')

# print("after removal of outliers")
# print("shape : ",df.shape)
# print("Class : ",df["Class"].unique())

def draw_scatterplot(ax, col1, col2, df):
    unique_labels = df.Class.unique()
    for cls in unique_labels:
        filtered = df[df.Class == cls]
        ax.scatter(filtered[col1], filtered[col2], label=cls, alpha=0.2)
    ax.legend()
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f'Joint scatterplot: {col1} & {col2}')

fig, ax = plt.subplots(3, 3, figsize=(15, 15))
pairs = [('Area', 'Eccentricity'), ('Eccentricity', 'Solidity'), ('Area', 'EquivDiameter'), ('roundness', 'Compactness'),
         ('ShapeFactor1', 'ShapeFactor2'), ('ShapeFactor2', 'ShapeFactor3'), ('ShapeFactor3', 'ShapeFactor4'), ('Compactness', 'Solidity'),
         ('Area', 'Solidity')]
for idx, p in enumerate(pairs):
    draw_scatterplot(ax[idx // 3, idx % 3], *p, df)
plt.tight_layout()
plt.savefig('./eda_outputs/scatterplot.png')
print("scatter plot saved....")