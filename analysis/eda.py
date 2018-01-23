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
credit.rename(columns={'default payment next month': 'default_next_m'})

# Exploratory plots
plt.figure()
g = sns.factorplot(x='EDUCATION', palette="Set3", y="default_next_m", data=credit, kind="bar")
#plt.savefig('/Users/Jostein/Grad School/SMU/7331/project1/credit-default/plots/dist_educ_default')