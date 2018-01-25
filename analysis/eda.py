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
corr = credit.corr(method='pearson')
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    plt.figure()
    credit_heatmap_corr = sns.heatmap(corr, mask=mask, square=True,
                                    linewidths=.1, cmap="YlGnBu")
    plt.savefig('/Users/Jostein/Grad School/SMU/7331/project1/credit-default/'
                + '/plots/corr_heatmap')
