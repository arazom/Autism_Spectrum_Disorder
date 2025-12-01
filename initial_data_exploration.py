#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
استكشاف أولي لبيانات التوحد
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# إعداد matplotlib للغة العربية
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# قراءة البيانات
file_path = '/home/ubuntu/upload/1CLSnewDatNOsum.xlsx'
df = pd.read_excel(file_path)

print("=" * 80)
print("معلومات أساسية عن مجموعة البيانات")
print("=" * 80)
print(f"\nعدد الصفوف (الأطفال): {df.shape[0]}")
print(f"عدد الأعمدة (المتغيرات): {df.shape[1]}")
print(f"\nأسماء الأعمدة:")
print(df.columns.tolist())

print("\n" + "=" * 80)
print("أول 5 صفوف من البيانات")
print("=" * 80)
print(df.head())

print("\n" + "=" * 80)
print("معلومات عن أنواع البيانات")
print("=" * 80)
print(df.info())

print("\n" + "=" * 80)
print("الإحصاءات الوصفية")
print("=" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("القيم المفقودة")
print("=" * 80)
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("لا توجد قيم مفقودة!")

print("\n" + "=" * 80)
print("توزيع مستويات الشدة")
print("=" * 80)
# البحث عن عمود مستوى الشدة
severity_cols = [col for col in df.columns if 'severity' in col.lower() or 'شدة' in col.lower() 
                 or 'level' in col.lower() or 'class' in col.lower() or 'label' in col.lower()
                 or 'مستوى' in col.lower()]

if severity_cols:
    severity_col = severity_cols[0]
    print(f"\nعمود مستوى الشدة: {severity_col}")
    print(f"\nتوزيع الفئات:")
    print(df[severity_col].value_counts().sort_index())
    print(f"\nالنسب المئوية:")
    print(df[severity_col].value_counts(normalize=True).sort_index() * 100)
else:
    print("لم يتم العثور على عمود مستوى الشدة!")
    print("\nالأعمدة المتاحة:")
    for col in df.columns:
        print(f"  - {col}: {df[col].nunique()} قيمة فريدة")

# حفظ النتائج
output_file = '/home/ubuntu/data_exploration_summary.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ملخص استكشاف البيانات\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"عدد الصفوف: {df.shape[0]}\n")
    f.write(f"عدد الأعمدة: {df.shape[1]}\n\n")
    f.write("أسماء الأعمدة:\n")
    for col in df.columns:
        f.write(f"  - {col}\n")
    
    if severity_cols:
        f.write(f"\n\nتوزيع مستويات الشدة ({severity_cols[0]}):\n")
        f.write(df[severity_cols[0]].value_counts().sort_index().to_string())

print(f"\n\nتم حفظ الملخص في: {output_file}")

