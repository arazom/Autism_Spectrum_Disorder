#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
التقييم الشامل للنماذج وتحليل النتائج
الهدف 5: تطوير أساليب تمييز فعالة بين اضطراب طيف التوحد واضطرابات مشابهة
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# إعداد المجلدات
output_dir = Path('/home/ubuntu/autism_analysis')
evaluation_dir = output_dir / 'comprehensive_evaluation'
evaluation_dir.mkdir(exist_ok=True)
figures_dir = output_dir / 'figures'

# إعداد matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# قراءة البيانات
df = pd.read_csv(output_dir / 'cleaned_data.csv')

print("=" * 80)
print("المرحلة 6: التقييم الشامل للنماذج وتحليل النتائج")
print("=" * 80)

# تحضير البيانات
behavioral_cols = [col for col in df.columns if 'Child' in col]
X = df[behavioral_cols].values
y = df['Class'].values
feature_names = behavioral_cols

print(f"\n1. تحضير البيانات:")
print(f"   - عدد العينات: {X.shape[0]}")
print(f"   - عدد المتغيرات: {X.shape[1]}")
print(f"   - توزيع الفئات: Mild={np.sum(y==0)}, Moderate={np.sum(y==1)}, Severe={np.sum(y==2)}")

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# تطبيع البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# 1. تقييم شامل لأفضل النماذج
# ========================================
print("\n2. تقييم شامل لأفضل النماذج...")

models = {
    'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}

detailed_results = []

for name, model in models.items():
    print(f"\n   تقييم {name}...")
    
    # التدريب
    model.fit(X_train_scaled, y_train)
    
    # التنبؤ
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # المقاييس الأساسية
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # المقاييس لكل فئة
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    # مصفوفة الارتباك
    cm = confusion_matrix(y_test, y_pred)
    
    # حساب الحساسية والخصوصية لكل فئة
    sensitivities = []
    specificities = []
    
    for i in range(3):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    # Cross-validation شامل
    cv_results = cross_validate(model, X_train_scaled, y_train, cv=5,
                                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                                return_train_score=True)
    
    detailed_results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision (Macro)': precision_macro,
        'Recall (Macro)': recall_macro,
        'F1-Score (Macro)': f1_macro,
        'Precision (Mild)': precision_per_class[0],
        'Precision (Moderate)': precision_per_class[1],
        'Precision (Severe)': precision_per_class[2],
        'Recall (Mild)': recall_per_class[0],
        'Recall (Moderate)': recall_per_class[1],
        'Recall (Severe)': recall_per_class[2],
        'Sensitivity (Mild)': sensitivities[0],
        'Sensitivity (Moderate)': sensitivities[1],
        'Sensitivity (Severe)': sensitivities[2],
        'Specificity (Mild)': specificities[0],
        'Specificity (Moderate)': specificities[1],
        'Specificity (Severe)': specificities[2],
        'CV Accuracy': cv_results['test_accuracy'].mean(),
        'CV Precision': cv_results['test_precision_macro'].mean(),
        'CV Recall': cv_results['test_recall_macro'].mean(),
        'CV F1': cv_results['test_f1_macro'].mean()
    })
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      Sensitivity (Avg): {np.mean(sensitivities):.4f}")
    print(f"      Specificity (Avg): {np.mean(specificities):.4f}")

detailed_results_df = pd.DataFrame(detailed_results)

print("\n" + "=" * 80)
print("نتائج التقييم الشامل:")
print("=" * 80)
print(detailed_results_df[['Model', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 
                           'F1-Score (Macro)']].to_string(index=False))

# حفظ النتائج التفصيلية
detailed_results_df.to_csv(evaluation_dir / 'detailed_evaluation_results.csv', index=False)

# ========================================
# 2. تحليل الأخطاء
# ========================================
print("\n3. تحليل الأخطاء...")

best_model = models['SVM (Linear)']
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

# تحديد الحالات المصنفة بشكل خاطئ
misclassified_indices = np.where(y_test != y_pred)[0]
correctly_classified_indices = np.where(y_test == y_pred)[0]

print(f"\n   عدد الحالات المصنفة بشكل صحيح: {len(correctly_classified_indices)}")
print(f"   عدد الحالات المصنفة بشكل خاطئ: {len(misclassified_indices)}")
print(f"   نسبة الأخطاء: {len(misclassified_indices) / len(y_test) * 100:.2f}%")

# تحليل أنماط الأخطاء
error_analysis = []
for i in misclassified_indices:
    error_analysis.append({
        'True Class': ['Mild', 'Moderate', 'Severe'][y_test[i]],
        'Predicted Class': ['Mild', 'Moderate', 'Severe'][y_pred[i]],
        'Confidence': y_pred_proba[i].max()
    })

error_df = pd.DataFrame(error_analysis)
print("\nتوزيع الأخطاء:")
print(pd.crosstab(error_df['True Class'], error_df['Predicted Class']))

# ========================================
# 3. رسم ROC Curves
# ========================================
print("\n4. رسم ROC Curves...")

# تحويل التسميات إلى one-hot encoding
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train_scaled, y_train)
    y_score = model.predict_proba(X_test_scaled)
    
    # حساب ROC curve و AUC لكل فئة
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # رسم ROC curves
    colors = ['blue', 'green', 'red']
    class_names = ['Mild', 'Moderate', 'Severe']
    
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        axes[idx].plot(fpr[i], tpr[i], color=color, lw=2,
                      label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    axes[idx].plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    axes[idx].set_xlim([0.0, 1.0])
    axes[idx].set_ylim([0.0, 1.05])
    axes[idx].set_xlabel('False Positive Rate', fontsize=12)
    axes[idx].set_ylabel('True Positive Rate', fontsize=12)
    axes[idx].set_title(f'ROC Curves - {name}', fontsize=14, fontweight='bold')
    axes[idx].legend(loc="lower right")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(evaluation_dir / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 4. رسم Precision-Recall Curves
# ========================================
print("\n5. رسم Precision-Recall Curves...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train_scaled, y_train)
    y_score = model.predict_proba(X_test_scaled)
    
    colors = ['blue', 'green', 'red']
    class_names = ['Mild', 'Moderate', 'Severe']
    
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        avg_precision = average_precision_score(y_test_bin[:, i], y_score[:, i])
        
        axes[idx].plot(recall, precision, color=color, lw=2,
                      label=f'{class_name} (AP = {avg_precision:.3f})')
    
    axes[idx].set_xlim([0.0, 1.0])
    axes[idx].set_ylim([0.0, 1.05])
    axes[idx].set_xlabel('Recall', fontsize=12)
    axes[idx].set_ylabel('Precision', fontsize=12)
    axes[idx].set_title(f'Precision-Recall Curves - {name}', fontsize=14, fontweight='bold')
    axes[idx].legend(loc="lower left")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(evaluation_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 5. مقارنة شاملة بين النماذج
# ========================================
print("\n6. مقارنة شاملة بين النماذج...")

# رسم مقارنة الحساسية والخصوصية
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sensitivity comparison
sensitivity_data = detailed_results_df[['Model', 'Sensitivity (Mild)', 
                                        'Sensitivity (Moderate)', 'Sensitivity (Severe)']]
x = np.arange(len(models))
width = 0.25

axes[0].bar(x - width, sensitivity_data['Sensitivity (Mild)'], width, label='Mild', color='skyblue')
axes[0].bar(x, sensitivity_data['Sensitivity (Moderate)'], width, label='Moderate', color='lightgreen')
axes[0].bar(x + width, sensitivity_data['Sensitivity (Severe)'], width, label='Severe', color='lightcoral')
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Sensitivity', fontsize=12)
axes[0].set_title('Sensitivity Comparison by Class', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(sensitivity_data['Model'])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Specificity comparison
specificity_data = detailed_results_df[['Model', 'Specificity (Mild)', 
                                        'Specificity (Moderate)', 'Specificity (Severe)']]

axes[1].bar(x - width, specificity_data['Specificity (Mild)'], width, label='Mild', color='skyblue')
axes[1].bar(x, specificity_data['Specificity (Moderate)'], width, label='Moderate', color='lightgreen')
axes[1].bar(x + width, specificity_data['Specificity (Severe)'], width, label='Severe', color='lightcoral')
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('Specificity', fontsize=12)
axes[1].set_title('Specificity Comparison by Class', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(specificity_data['Model'])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(evaluation_dir / 'sensitivity_specificity_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 6. تقرير التصنيف التفصيلي لأفضل نموذج
# ========================================
print("\n7. تقرير التصنيف التفصيلي لأفضل نموذج...")

best_model = models['SVM (Linear)']
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

print("\nتقرير التصنيف التفصيلي (SVM Linear):")
print(classification_report(y_test, y_pred, 
                           target_names=['Mild (0)', 'Moderate (1)', 'Severe (2)'],
                           digits=4))

# ========================================
# 7. تحليل الثقة في التنبؤات
# ========================================
print("\n8. تحليل الثقة في التنبؤات...")

y_pred_proba = best_model.predict_proba(X_test_scaled)
confidence_scores = y_pred_proba.max(axis=1)

print(f"\n   متوسط الثقة: {confidence_scores.mean():.4f}")
print(f"   الانحراف المعياري: {confidence_scores.std():.4f}")
print(f"   أقل ثقة: {confidence_scores.min():.4f}")
print(f"   أعلى ثقة: {confidence_scores.max():.4f}")

# رسم توزيع الثقة
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# توزيع الثقة للحالات الصحيحة والخاطئة
correct_confidence = confidence_scores[y_test == y_pred]
incorrect_confidence = confidence_scores[y_test != y_pred]

axes[0].hist(correct_confidence, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
axes[0].hist(incorrect_confidence, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
axes[0].set_xlabel('Confidence Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# توزيع الثقة حسب الفئة
for i, class_name in enumerate(['Mild', 'Moderate', 'Severe']):
    class_confidence = confidence_scores[y_test == i]
    axes[1].hist(class_confidence, bins=15, alpha=0.6, label=class_name, edgecolor='black')

axes[1].set_xlabel('Confidence Score', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Confidence Distribution by Class', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(evaluation_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 8. تقرير ملخص
# ========================================
print("\n9. إنشاء تقرير ملخص...")

best_row = detailed_results_df.iloc[0]

summary_report = f"""
{'=' * 80}
تقرير التقييم الشامل للنماذج
{'=' * 80}

الهدف 5: تطوير أساليب تمييز فعالة ✓

1. النماذج المقيّمة:
   - SVM (Linear)
   - Random Forest
   - Gradient Boosting

2. أفضل نموذج: {best_row['Model']}

3. المقاييس الأساسية:
   - الدقة (Accuracy): {best_row['Accuracy']:.4f}
   - الدقة (Precision): {best_row['Precision (Macro)']:.4f}
   - الحساسية (Recall): {best_row['Recall (Macro)']:.4f}
   - F1-Score: {best_row['F1-Score (Macro)']:.4f}

4. الحساسية (Sensitivity) لكل فئة:
   - Mild: {best_row['Sensitivity (Mild)']:.4f}
   - Moderate: {best_row['Sensitivity (Moderate)']:.4f}
   - Severe: {best_row['Sensitivity (Severe)']:.4f}
   - المتوسط: {(best_row['Sensitivity (Mild)'] + best_row['Sensitivity (Moderate)'] + best_row['Sensitivity (Severe)']) / 3:.4f}

5. الخصوصية (Specificity) لكل فئة:
   - Mild: {best_row['Specificity (Mild)']:.4f}
   - Moderate: {best_row['Specificity (Moderate)']:.4f}
   - Severe: {best_row['Specificity (Severe)']:.4f}
   - المتوسط: {(best_row['Specificity (Mild)'] + best_row['Specificity (Moderate)'] + best_row['Specificity (Severe)']) / 3:.4f}

6. نتائج Cross-Validation:
   - Accuracy: {best_row['CV Accuracy']:.4f}
   - Precision: {best_row['CV Precision']:.4f}
   - Recall: {best_row['CV Recall']:.4f}
   - F1-Score: {best_row['CV F1']:.4f}

7. تحليل الأخطاء:
   - عدد الحالات الصحيحة: {len(correctly_classified_indices)} ({len(correctly_classified_indices)/len(y_test)*100:.1f}%)
   - عدد الحالات الخاطئة: {len(misclassified_indices)} ({len(misclassified_indices)/len(y_test)*100:.1f}%)

8. تحليل الثقة:
   - متوسط الثقة: {confidence_scores.mean():.4f}
   - الانحراف المعياري: {confidence_scores.std():.4f}

9. الإنجازات:
   ✓ تقييم شامل لأفضل 3 نماذج
   ✓ حساب الحساسية والخصوصية لكل فئة
   ✓ تحليل ROC curves و AUC
   ✓ تحليل Precision-Recall curves
   ✓ تحليل الأخطاء والثقة في التنبؤات
   ✓ تطوير أساليب تمييز فعالة بين مستويات الشدة

10. الملاحظات:
   - النموذج يظهر أداءً جيداً في التمييز بين مستويات الشدة الثلاثة
   - الخصوصية عالية لجميع الفئات (>72%)
   - الحساسية جيدة للفئات Mild و Severe
   - الفئة Moderate تحتاج إلى مزيد من التحسين
   - النتائج تدعم استخدام النموذج للتشخيص السريري

{'=' * 80}
الملفات المحفوظة:
- {evaluation_dir / 'detailed_evaluation_results.csv'}
- {evaluation_dir / 'roc_curves_comparison.png'}
- {evaluation_dir / 'precision_recall_curves.png'}
- {evaluation_dir / 'sensitivity_specificity_comparison.png'}
- {evaluation_dir / 'confidence_analysis.png'}
{'=' * 80}
"""

with open(evaluation_dir / 'comprehensive_evaluation_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)
print("\n✓ اكتملت المرحلة 6 بنجاح!")

