#!/usr/bin/env python3
"""
Debug script to check CSV column names
"""

import pandas as pd

# Read the CSV file
df = pd.read_csv('data/customer.csv')

print("Original column names:")
print(list(df.columns))
print()

print("Column names after strip('\"'):")
df_strip = df.copy()
df_strip.columns = df_strip.columns.str.strip('"')
print(list(df_strip.columns))
print()

print("Column names after replace('\"', ''):")
df_replace = df.copy()
df_replace.columns = df_replace.columns.str.replace('"', '')
print(list(df_replace.columns))
print()

print("First few rows:")
print(df.head())
