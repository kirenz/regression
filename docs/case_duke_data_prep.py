#!/usr/bin/env python
# coding: utf-8

import pandas as pd

ROOT = "https://raw.githubusercontent.com/kirenz/modern-statistics/main/data/"
DATA = "duke-forest.csv"

df = pd.read_csv(ROOT + DATA)

# drop column with too many missing values
df = df.drop(['hoa'], axis=1)

# drop remaining row with one missing value
df = df.dropna()

# Drop irrelevant features
df = df.drop(['url', 'address'], axis=1)

# Convert data types
categorical_list = ['type', 'heating', 'cooling', 'parking']

for i in categorical_list:
    df[i] = df[i].astype("category")

# drop irrelavant columns
df = df.drop(['type', 'heating', 'parking'], axis=1)