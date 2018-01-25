#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:30:04 2018

@author: Jostein
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import statistics as st
import csv as csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

filepath = "/Users/Jostein/Grad School/SMU/7331/project1/credit-default/data/default of credit card clients.xls"
credit = pd.read_excel(filepath, header=1, skiprows=0)

# Rename column(s)
credit = credit.rename(columns={'default payment next month': 'default_next_m'})
credit.columns

# Exploratory plots
# Count plot of credit default by sex
plt.figure()
g = sns.factorplot(hue='SEX', palette="Set3", y="default_next_m", data=credit, kind="count")
plt.savefig('/Users/Jostein/Grad School/SMU/7331/project1/credit-default/plots/dist_sex_default')

# Count plot of credit default by education
plt.figure()
g = sns.factorplot(hue='EDUCATION', palette="Set3", y="default_next_m", data=credit, kind="count")
plt.savefig('/Users/Jostein/Grad School/SMU/7331/project1/credit-default/plots/dist_education_default')

# Different cubehelix palettes
plt.figure()
x = credit.AGE
y = credit.EDUCATION
ax = sns.kdeplot(x, y, cbar=True, shade=True)
plt.savefig('/Users/Jostein/Grad School/SMU/7331/project1/credit-default/plots/kde_age_educ')

sns.set(style="dark")
#rs = np.random.RandomState(50)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

# Rotate the starting point around the cubehelix hue circle
for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):

    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    x, y = rs.randn(2, 50)
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=ax)
    ax.set(xlim=(-3, 3), ylim=(-3, 3))

f.tight_layout()

# Pretty correlation heat map matrix
# Source:
# http://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap
corr = credit.corr(method='pearson')
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    plt.figure()
    credit_heatmap_corr = sns.heatmap(corr, mask=mask, square=True,
                                    linewidths=.1, cmap="YlGnBu")
    plt.savefig('/Users/Jostein/Grad School/SMU/7331/project1/credit-default/'
                + '/plots/corr_heatmap')
    

# Cluster Heat Map
# Source:
# http://seaborn.pydata.org/generated/seaborn.clustermap.html
plt.figure()
sns.set(color_codes=True)
#y_defaults = credit.pop("default_next_m")
y_defaults = credit.default_next_m
lut = dict(zip(y_defaults.unique(), "bg"))
row_colors = y_defaults.map(lut)
g = sns.clustermap(credit, row_colors=row_colors, standard_scale=1)
g.savefig('/Users/Jostein/Grad School/SMU/7331/project1/credit-default/'
                + '/plots/cluster_heatmap')

# Violin plot of limit balance distribution by education level per sex
# Source:
# http://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot
plt.figure()
ax = sns.violinplot(x="EDUCATION", y="LIMIT_BAL", hue="SEX", data=credit, split=True)
plt.savefig('/Users/Jostein/Grad School/SMU/7331/project1/credit-default/'
                + '/plots/violin-limitbalance-by-education-per-sex')

# Build the logistic regression
# Source:
# https://machinelearningmastery.com/feature-selection-machine-learning-python/

# Find total number of columns
len(credit.columns)

# Set X variables for the model
# Slices all attributes except for 'default_next_m' into X
X = credit[credit.columns[:24]]

# Set Y variable for the model
Y = credit['default_next_m']

# Recursive Feature Selection for the top 10 performing variables
model = LogisticRegression()
rfe = RFE(model, 10)
fit = rfe.fit(X, Y)

print(fit)
print("Number of Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


