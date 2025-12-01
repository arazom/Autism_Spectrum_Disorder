#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المعالجة المسبقة والتحليل الاستكشافي الشامل لبيانات التوحد
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# إنشاء مجلد للنتائج
output_dir = Path('/home/ubuntu/autism_analysis')
output_dir.mkdir(exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# إعداد matplotlib للغة العربية
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# قراءة البيانات
df = pd.read_excel('/home/ubuntu/upload/1CLSnewDatNOsum.xlsx')

print("=" * 80)
print("المرحلة 2: المعالجة المسبقة والتحليل الاستكشافي")
print("=" * 80)

# 1. تنظيف أسماء الأعمدة
print("\n1. تنظيف أسماء الأعمدة...")
df.columns = df.columns.str.strip()
print(f"   تم تنظيف {len(df.columns)} عمود")

# 2. فصل المتغيرات
demographic_cols = ['Gender', 'Age', 'Fam_Hist', 'No_OfChild', 'Seq_ofChild', 
                    'Father_Emp', 'Mother_Emp', 'Blood_Relat']
behavioral_cols = [col for col in df.columns if 'Child' in col]
target_col = 'Class'

print(f"\n2. تصنيف المتغيرات:")
print(f"   - متغيرات ديموغرافية: {len(demographic_cols)}")
print(f"   - متغيرات سلوكية: {len(behavioral_cols)}")
print(f"   - متغير الهدف: {target_col}")

# 3. التحليل الإحصائي الوصفي
print("\n3. التحليل الإحصائي الوصفي...")

# توزيع العمر حسب مستوى الشدة
age_by_class = df.groupby('Class')['Age'].describe()
print("\nتوزيع العمر حسب مستوى الشدة:")
print(age_by_class)

# توزيع الجنس حسب مستوى الشدة
gender_by_class = pd.crosstab(df['Class'], df['Gender'], normalize='index') * 100
print("\nتوزيع الجنس حسب مستوى الشدة (%):")
print(gender_by_class)

# 4. تحليل الارتباط
print("\n4. تحليل الارتباط...")
correlation_matrix = df[behavioral_cols + [target_col]].corr()

# حفظ مصفوفة الارتباط
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Behavioral Features', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(figures_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   تم حفظ مصفوفة الارتباط")

# أهم الارتباطات مع متغير الهدف
target_correlations = correlation_matrix[target_col].drop(target_col).sort_values(ascending=False)
print("\nأعلى 10 ارتباطات مع مستوى الشدة:")
print(target_correlations.head(10))
print("\nأقل 10 ارتباطات مع مستوى الشدة:")
print(target_correlations.tail(10))

# 5. تحليل توزيع المتغيرات السلوكية
print("\n5. تحليل توزيع المتغيرات السلوكية...")

# رسم توزيعات المتغيرات السلوكية
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.ravel()

for idx, col in enumerate(behavioral_cols):
    if idx < len(axes):
        df[col].hist(bins=5, ax=axes[idx], edgecolor='black')
        axes[idx].set_title(col.replace('Child', '').replace('_', ' '), fontsize=8)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(figures_dir / 'behavioral_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   تم حفظ توزيعات المتغيرات السلوكية")

# 6. تحليل الفروق بين المستويات
print("\n6. تحليل الفروق بين مستويات الشدة...")

# اختبار ANOVA لكل متغير سلوكي
anova_results = []
for col in behavioral_cols:
    groups = [df[df['Class'] == i][col].values for i in [0, 1, 2]]
    f_stat, p_value = stats.f_oneway(*groups)
    anova_results.append({
        'Feature': col,
        'F-statistic': f_stat,
        'p-value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

anova_df = pd.DataFrame(anova_results).sort_values('F-statistic', ascending=False)
print("\nنتائج اختبار ANOVA (أعلى 10 متغيرات):")
print(anova_df.head(10))

# حفظ النتائج
anova_df.to_csv(output_dir / 'anova_results.csv', index=False)

# 7. رسم box plots لأهم المتغيرات
print("\n7. رسم box plots لأهم المتغيرات...")
top_features = anova_df.head(12)['Feature'].tolist()

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.ravel()

for idx, col in enumerate(top_features):
    if idx < len(axes):
        df.boxplot(column=col, by='Class', ax=axes[idx])
        axes[idx].set_title(col.replace('Child', '').replace('_', ' '))
        axes[idx].set_xlabel('Severity Level')
        axes[idx].set_ylabel('Value')
        plt.sca(axes[idx])
        plt.xticks([1, 2, 3], ['Mild (0)', 'Moderate (1)', 'Severe (2)'])

plt.suptitle('')
plt.tight_layout()
plt.savefig(figures_dir / 'top_features_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   تم حفظ box plots")

# 8. تحليل المكونات الرئيسية (PCA) للتصور
print("\n8. تطبيق PCA للتصور...")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df[behavioral_cols].values
y = df[target_col].values

# تطبيع البيانات
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تطبيق PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"   نسبة التباين المفسر بواسطة المكونين الأولين: {pca.explained_variance_ratio_.sum():.2%}")

# رسم PCA
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                     alpha=0.6, edgecolors='black', s=100)
plt.colorbar(scatter, label='Severity Level')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Visualization of Autism Dataset', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'pca_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   تم حفظ تصور PCA")

# 9. حفظ البيانات المعالجة
print("\n9. حفظ البيانات المعالجة...")
df.to_csv(output_dir / 'cleaned_data.csv', index=False)
print(f"   تم حفظ البيانات في: {output_dir / 'cleaned_data.csv'}")

# 10. إنشاء تقرير ملخص
print("\n10. إنشاء تقرير ملخص...")
summary_report = f"""
{'=' * 80}
تقرير المعالجة المسبقة والتحليل الاستكشافي
{'=' * 80}

1. معلومات البيانات:
   - عدد العينات: {df.shape[0]}
   - عدد المتغيرات: {df.shape[1]}
   - متغيرات ديموغرافية: {len(demographic_cols)}
   - متغيرات سلوكية: {len(behavioral_cols)}

2. توزيع مستويات الشدة:
   - المستوى 0 (خفيف): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)
   - المستوى 1 (متوسط): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)
   - المستوى 2 (عالي): {(y == 2).sum()} ({(y == 2).sum() / len(y) * 100:.1f}%)

3. توزيع العمر:
   - المتوسط: {df['Age'].mean():.2f} سنة
   - الانحراف المعياري: {df['Age'].std():.2f}
   - المدى: {df['Age'].min()}-{df['Age'].max()} سنة

4. توزيع الجنس:
   - ذكور: {(df['Gender'] == 1).sum()} ({(df['Gender'] == 1).sum() / len(df) * 100:.1f}%)
   - إناث: {(df['Gender'] == 0).sum()} ({(df['Gender'] == 0).sum() / len(df) * 100:.1f}%)

5. أهم المتغيرات المرتبطة بمستوى الشدة:
{target_correlations.head(10).to_string()}

6. عدد المتغيرات ذات الفروق الدالة إحصائياً (p < 0.05):
   {(anova_df['p-value'] < 0.05).sum()} من أصل {len(behavioral_cols)}

7. نسبة التباين المفسر بواسطة أول مكونين رئيسيين:
   {pca.explained_variance_ratio_.sum():.2%}

{'=' * 80}
الملفات المحفوظة:
- {output_dir / 'cleaned_data.csv'}
- {output_dir / 'anova_results.csv'}
- {figures_dir / 'correlation_matrix.png'}
- {figures_dir / 'behavioral_distributions.png'}
- {figures_dir / 'top_features_boxplots.png'}
- {figures_dir / 'pca_visualization.png'}
{'=' * 80}
"""

with open(output_dir / 'preprocessing_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)
print("\n✓ اكتملت المرحلة 2 بنجاح!")

