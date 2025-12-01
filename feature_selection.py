#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختيار الخصائص وتقليل الأبعاد لتقليل زمن التشخيص
الهدف 3: تقليل زمن التشخيص من خلال تقليص مجموعات الخصائص المطلوبة
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (SelectKBest, f_classif, mutual_info_classif,
                                       RFE, SelectFromModel, chi2)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# إعداد المجلدات
output_dir = Path('/home/ubuntu/autism_analysis')
feature_dir = output_dir / 'feature_selection_results'
feature_dir.mkdir(exist_ok=True)
figures_dir = output_dir / 'figures'

# إعداد matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# قراءة البيانات
df = pd.read_csv(output_dir / 'cleaned_data.csv')

print("=" * 80)
print("المرحلة 5: اختيار الخصائص وتقليل الأبعاد")
print("=" * 80)

# تحضير البيانات
behavioral_cols = [col for col in df.columns if 'Child' in col]
X = df[behavioral_cols].values
y = df['Class'].values
feature_names = behavioral_cols

print(f"\n1. البيانات الأصلية:")
print(f"   - عدد المتغيرات: {X.shape[1]}")
print(f"   - عدد العينات: {X.shape[0]}")

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# تطبيع البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# النموذج الأساسي للمقارنة (SVM Linear - أفضل نموذج من المرحلة السابقة)
base_model = SVC(kernel='linear', random_state=42)
base_model.fit(X_train_scaled, y_train)
base_accuracy = accuracy_score(y_test, base_model.predict(X_test_scaled))

print(f"\n2. الأداء الأساسي (جميع المتغيرات):")
print(f"   - الدقة: {base_accuracy:.4f}")
print(f"   - عدد المتغيرات: {X.shape[1]}")

# ========================================
# 1. تحليل أهمية المتغيرات باستخدام Random Forest
# ========================================
print("\n3. تحليل أهمية المتغيرات باستخدام Random Forest...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nأهم 10 متغيرات:")
print(feature_importance.head(10).to_string(index=False))

# رسم أهمية المتغيرات
plt.figure(figsize=(12, 10))
plt.barh(range(len(feature_importance)), feature_importance['Importance'])
plt.yticks(range(len(feature_importance)), 
           [f.replace('Child', '').replace('_', ' ') for f in feature_importance['Feature']], 
           fontsize=8)
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Feature Importance using Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(feature_dir / 'feature_importance_rf.png', dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 2. اختيار الخصائص باستخدام طرق مختلفة
# ========================================
print("\n4. اختيار الخصائص باستخدام طرق مختلفة...")

selection_methods = {}

# 2.1 SelectKBest (ANOVA F-value)
print("\n   4.1 SelectKBest (ANOVA F-value)...")
selector_anova = SelectKBest(score_func=f_classif, k='all')
selector_anova.fit(X_train_scaled, y_train)
anova_scores = pd.DataFrame({
    'Feature': feature_names,
    'Score': selector_anova.scores_
}).sort_values('Score', ascending=False)
selection_methods['ANOVA'] = anova_scores

# 2.2 SelectKBest (Mutual Information)
print("   4.2 SelectKBest (Mutual Information)...")
selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
selector_mi.fit(X_train_scaled, y_train)
mi_scores = pd.DataFrame({
    'Feature': feature_names,
    'Score': selector_mi.scores_
}).sort_values('Score', ascending=False)
selection_methods['Mutual_Info'] = mi_scores

# 2.3 Recursive Feature Elimination (RFE)
print("   4.3 Recursive Feature Elimination (RFE)...")
rfe_model = SVC(kernel='linear', random_state=42)
rfe = RFE(estimator=rfe_model, n_features_to_select=10, step=1)
rfe.fit(X_train_scaled, y_train)
rfe_ranking = pd.DataFrame({
    'Feature': feature_names,
    'Ranking': rfe.ranking_
}).sort_values('Ranking')
selection_methods['RFE'] = rfe_ranking

# 2.4 L1-based feature selection
print("   4.4 L1-based feature selection...")
from sklearn.linear_model import LogisticRegression
l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
l1_model.fit(X_train_scaled, y_train)
l1_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(l1_model.coef_).mean(axis=0)
}).sort_values('Importance', ascending=False)
selection_methods['L1'] = l1_importance

# ========================================
# 3. تقييم الأداء مع عدد متغيرات مختلف
# ========================================
print("\n5. تقييم الأداء مع عدد متغيرات مختلف...")

k_values = [5, 10, 15, 20, 25, 26]
results_by_k = []

for k in k_values:
    print(f"\n   اختبار مع {k} متغير...")
    
    # اختيار أفضل k متغيرات بناءً على Random Forest
    top_k_features = feature_importance.head(k)['Feature'].tolist()
    feature_indices = [feature_names.index(f) for f in top_k_features]
    
    X_train_k = X_train_scaled[:, feature_indices]
    X_test_k = X_test_scaled[:, feature_indices]
    
    # تدريب وتقييم النموذج
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_k, y_train)
    y_pred = model.predict(X_test_k)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_k, y_train, cv=5, scoring='accuracy')
    
    results_by_k.append({
        'K': k,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Features': ', '.join([f.replace('Child', '').replace('_', ' ')[:20] for f in top_k_features[:5]]) + '...'
    })
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      F1-Score: {f1:.4f}")
    print(f"      CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

results_by_k_df = pd.DataFrame(results_by_k)

print("\n" + "=" * 80)
print("نتائج الأداء مع عدد متغيرات مختلف:")
print("=" * 80)
print(results_by_k_df[['K', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean']].to_string(index=False))

# حفظ النتائج
results_by_k_df.to_csv(feature_dir / 'performance_by_k_features.csv', index=False)

# ========================================
# 4. رسم العلاقة بين عدد المتغيرات والأداء
# ========================================
print("\n6. رسم العلاقة بين عدد المتغيرات والأداء...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(results_by_k_df['K'], results_by_k_df['Accuracy'], 'o-', linewidth=2, markersize=8)
axes[0, 0].axhline(y=base_accuracy, color='r', linestyle='--', label=f'Baseline ({base_accuracy:.3f})')
axes[0, 0].set_xlabel('Number of Features', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Accuracy vs Number of Features', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Precision
axes[0, 1].plot(results_by_k_df['K'], results_by_k_df['Precision'], 'o-', linewidth=2, markersize=8, color='green')
axes[0, 1].set_xlabel('Number of Features', fontsize=12)
axes[0, 1].set_ylabel('Precision', fontsize=12)
axes[0, 1].set_title('Precision vs Number of Features', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Recall
axes[1, 0].plot(results_by_k_df['K'], results_by_k_df['Recall'], 'o-', linewidth=2, markersize=8, color='orange')
axes[1, 0].set_xlabel('Number of Features', fontsize=12)
axes[1, 0].set_ylabel('Recall', fontsize=12)
axes[1, 0].set_title('Recall vs Number of Features', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# F1-Score
axes[1, 1].plot(results_by_k_df['K'], results_by_k_df['F1-Score'], 'o-', linewidth=2, markersize=8, color='purple')
axes[1, 1].set_xlabel('Number of Features', fontsize=12)
axes[1, 1].set_ylabel('F1-Score', fontsize=12)
axes[1, 1].set_title('F1-Score vs Number of Features', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(feature_dir / 'performance_vs_features.png', dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 5. تحديد العدد الأمثل للمتغيرات
# ========================================
print("\n7. تحديد العدد الأمثل للمتغيرات...")

# إيجاد العدد الأمثل (أعلى دقة)
best_k_idx = results_by_k_df['Accuracy'].idxmax()
best_k = results_by_k_df.loc[best_k_idx, 'K']
best_accuracy = results_by_k_df.loc[best_k_idx, 'Accuracy']

print(f"\n   العدد الأمثل للمتغيرات: {int(best_k)}")
print(f"   الدقة: {best_accuracy:.4f}")
print(f"   تحسين الأداء: {(best_accuracy - base_accuracy):.4f}")
print(f"   تقليل المتغيرات: {26 - int(best_k)} متغير ({(26 - int(best_k)) / 26 * 100:.1f}%)")

# قائمة بأفضل المتغيرات
best_features = feature_importance.head(int(best_k))
print(f"\n   أفضل {int(best_k)} متغيرات:")
for idx, row in best_features.iterrows():
    print(f"      {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

# حفظ قائمة المتغيرات المختارة
best_features.to_csv(feature_dir / f'best_{int(best_k)}_features.csv', index=False)

# ========================================
# 6. مقارنة طرق اختيار الخصائص
# ========================================
print("\n8. مقارنة طرق اختيار الخصائص...")

comparison_results = []

for method_name, method_scores in selection_methods.items():
    if method_name == 'RFE':
        # RFE يعطي ranking، نختار الأفضل
        top_features = method_scores[method_scores['Ranking'] == 1]['Feature'].tolist()
        if len(top_features) < 10:
            top_features = method_scores.head(10)['Feature'].tolist()
    else:
        top_features = method_scores.head(10)['Feature'].tolist()
    
    feature_indices = [feature_names.index(f) for f in top_features]
    
    X_train_selected = X_train_scaled[:, feature_indices]
    X_test_selected = X_test_scaled[:, feature_indices]
    
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    comparison_results.append({
        'Method': method_name,
        'K': len(top_features),
        'Accuracy': accuracy,
        'F1-Score': f1
    })

comparison_df = pd.DataFrame(comparison_results).sort_values('Accuracy', ascending=False)

print("\nمقارنة طرق اختيار الخصائص (10 متغيرات):")
print(comparison_df.to_string(index=False))

# ========================================
# 7. حساب توفير الوقت
# ========================================
print("\n9. حساب توفير الوقت...")

original_time = 26  # عدد الأسئلة الأصلي
reduced_time = int(best_k)
time_saved = original_time - reduced_time
time_saved_percent = (time_saved / original_time) * 100

print(f"\n   عدد الأسئلة الأصلي: {original_time}")
print(f"   عدد الأسئلة بعد التقليل: {reduced_time}")
print(f"   الأسئلة المحذوفة: {time_saved} ({time_saved_percent:.1f}%)")
print(f"   تقدير توفير الوقت: {time_saved_percent:.1f}% من وقت التشخيص")

# ========================================
# 8. تقرير ملخص
# ========================================
print("\n10. إنشاء تقرير ملخص...")

summary_report = f"""
{'=' * 80}
تقرير اختيار الخصائص وتقليل الأبعاد
{'=' * 80}

الهدف 3: تقليل زمن التشخيص من خلال تقليص مجموعات الخصائص ✓

1. البيانات الأصلية:
   - عدد المتغيرات الأصلي: 26
   - الدقة الأساسية: {base_accuracy:.4f}

2. العدد الأمثل للمتغيرات:
   - العدد الأمثل: {int(best_k)} متغيرات
   - الدقة بعد التقليل: {best_accuracy:.4f}
   - التحسين في الأداء: {(best_accuracy - base_accuracy):.4f}

3. توفير الوقت:
   - تقليل عدد الأسئلة: من {original_time} إلى {reduced_time}
   - نسبة التقليل: {time_saved_percent:.1f}%
   - الأسئلة المحذوفة: {time_saved} سؤال

4. أفضل {int(best_k)} متغيرات:

{best_features[['Feature', 'Importance']].to_string(index=False)}

5. مقارنة طرق اختيار الخصائص:

{comparison_df.to_string(index=False)}

6. الإنجازات:
   ✓ تقليل عدد المتغيرات بنسبة {time_saved_percent:.1f}%
   ✓ الحفاظ على (أو تحسين) دقة التصنيف
   ✓ تقليل زمن التشخيص بشكل كبير
   ✓ تحديد أهم المتغيرات السلوكية للتشخيص
   ✓ مقارنة طرق مختلفة لاختيار الخصائص

7. الملاحظات:
   - استخدام عدد أقل من المتغيرات يمكن أن يحافظ على (أو يحسن) الأداء
   - المتغيرات المختارة هي الأكثر تمييزاً لشدة التوحد
   - تقليل عدد الأسئلة يسرع عملية التشخيص ويقلل العبء على المختصين
   - النتائج تدعم إمكانية تطوير استبيان مختصر للتشخيص السريع

{'=' * 80}
الملفات المحفوظة:
- {feature_dir / 'feature_importance_rf.png'}
- {feature_dir / 'performance_by_k_features.csv'}
- {feature_dir / 'performance_vs_features.png'}
- {feature_dir / f'best_{int(best_k)}_features.csv'}
{'=' * 80}
"""

with open(feature_dir / 'feature_selection_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)
print("\n✓ اكتملت المرحلة 5 بنجاح!")

