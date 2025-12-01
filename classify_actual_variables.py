import pandas as pd
import numpy as np

# قراءة البيانات
df = pd.read_excel('/home/ubuntu/upload/1CLSnewDatNOsum.xlsx')

# المتغيرات الديموغرافية
demographic_vars = ['Gender', 'Age', 'Fam_Hist', 'No_OfChild', 'Seq_ofChild', 
                    'Father_Emp', 'Mother_Emp', 'Blood_Relat']

# استخراج المتغيرات السلوكية الفعلية
all_columns = df.columns.tolist()
behavioral_vars = [col for col in all_columns 
                   if col not in demographic_vars and col != 'Class']

print("=" * 100)
print("تصنيف المتغيرات السلوكية الـ 26 حسب المجالات الخمسة")
print("=" * 100)

# تصنيف المتغيرات الفعلية حسب المجالات
classification = {
    '1childName_response': 'التفاعل الاجتماعي',
    '2ChildEye_contact': 'التفاعل الاجتماعي',
    '3ChildObject_lining': 'السلوكيات النمطية والمتكررة',
    '4ChildSpeech_clarity': 'التواصل',
    '5ChildRequest_pointing': 'التفاعل الاجتماعي',
    '6ChildJoint_attention': 'التفاعل الاجتماعي',
    '7ChildRestriced_interests\n': 'السلوكيات النمطية والمتكررة',
    '8ChildVocabulary_size': 'التطور اللغوي',
    '9ChildSymbolic_play': 'التفاعل الاجتماعي',
    '10ChildShared_attention': 'التفاعل الاجتماعي',
    '11ChildSensory_seeking': 'الحساسية الحسية',
    '12ChildHand_prompting': 'التفاعل الاجتماعي',
    '13Child_walkONtiptoe': 'السلوكيات النمطية والمتكررة',
    '14ChildAdaptability_Flexibility': 'السلوكيات النمطية والمتكررة',
    '15Child_empathy': 'التفاعل الاجتماعي',
    '16Child_Repetitive_behaviors': 'السلوكيات النمطية والمتكررة',
    '17ChildFirst_words\n': 'التطور اللغوي',
    '18ChildEchoin_voice': 'التواصل',
    '19ChildGesture_use': 'التواصل',
    '20ChildFinger_move': 'السلوكيات النمطية والمتكررة',
    '21ChildChecks_reactions': 'التفاعل الاجتماعي',
    '22ChildFocused_attention': 'التفاعل الاجتماعي',
    '23ChildRepetitive_manipulation': 'السلوكيات النمطية والمتكررة',
    '24ChildNoise_sensitivity': 'الحساسية الحسية',
    '25ChildBlank_staring': 'السلوكيات النمطية والمتكررة',
}

# إنشاء قائمة بالنتائج
results = []

for var in behavioral_vars:
    if var in classification:
        domain = classification[var]
        correlation = df[var].corr(df['Class'])
        mean_val = df[var].mean()
        std_val = df[var].std()
        
        # تنظيف اسم المتغير
        clean_name = var.replace('Child', '').replace('child', '').replace('\n', '').replace('_', ' ').strip()
        
        results.append({
            'الرقم': var.split('Child')[0] if 'Child' in var or 'child' in var else '',
            'اسم المتغير': clean_name,
            'المجال': domain,
            'الارتباط مع الشدة': correlation,
            'المتوسط': mean_val,
            'الانحراف المعياري': std_val
        })

# إنشاء DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('الرقم')

# طباعة النتائج حسب المجال
domains_order = [
    'التفاعل الاجتماعي',
    'التواصل',
    'السلوكيات النمطية والمتكررة',
    'الحساسية الحسية',
    'التطور اللغوي'
]

for domain in domains_order:
    domain_vars = results_df[results_df['المجال'] == domain]
    
    print(f"\n{'='*100}")
    print(f"المجال: {domain}")
    print(f"{'='*100}")
    print(f"عدد المتغيرات: {len(domain_vars)}\n")
    
    for idx, row in domain_vars.iterrows():
        print(f"{row['الرقم']}. {row['اسم المتغير']}")
        print(f"   - الارتباط مع الشدة: {row['الارتباط مع الشدة']:.3f}")
        print(f"   - المتوسط: {row['المتوسط']:.2f}, الانحراف المعياري: {row['الانحراف المعياري']:.2f}")
        print()

# إحصائيات عامة
print("\n" + "="*100)
print("ملخص إحصائي حسب المجالات:")
print("="*100)

summary = results_df.groupby('المجال').agg({
    'اسم المتغير': 'count',
    'الارتباط مع الشدة': 'mean'
}).rename(columns={
    'اسم المتغير': 'عدد المتغيرات',
    'الارتباط مع الشدة': 'متوسط الارتباط'
})

# إعادة ترتيب حسب الترتيب المطلوب
summary = summary.reindex(domains_order)

print(summary)

# حفظ النتائج
results_df.to_csv('/home/ubuntu/autism_analysis/behavioral_variables_classification.csv', 
                  index=False, encoding='utf-8-sig')

# إنشاء جدول مفصل للتقرير
print("\n" + "="*100)
print("جدول شامل لجميع المتغيرات:")
print("="*100)

# ترتيب حسب المجال ثم الرقم
results_df_sorted = results_df.copy()
results_df_sorted['domain_order'] = results_df_sorted['المجال'].map({
    'التفاعل الاجتماعي': 1,
    'التواصل': 2,
    'السلوكيات النمطية والمتكررة': 3,
    'الحساسية الحسية': 4,
    'التطور اللغوي': 5
})
results_df_sorted = results_df_sorted.sort_values(['domain_order', 'الرقم'])

# طباعة الجدول
for idx, row in results_df_sorted.iterrows():
    print(f"{row['الرقم']:3s} | {row['اسم المتغير']:35s} | {row['المجال']:30s} | {row['الارتباط مع الشدة']:6.3f}")

print("\n" + "="*100)
print(f"تم حفظ التصنيف في: /home/ubuntu/autism_analysis/behavioral_variables_classification.csv")
print("="*100)

# إنشاء جدول للوورد/البحث
print("\n" + "="*100)
print("جدول للنسخ إلى البحث (Markdown format):")
print("="*100)

print("\n| الرقم | اسم المتغير | المجال | الارتباط مع الشدة |")
print("|------|-------------|--------|-------------------|")

for idx, row in results_df_sorted.iterrows():
    print(f"| {row['الرقم']} | {row['اسم المتغير']} | {row['المجال']} | {row['الارتباط مع الشدة']:.3f} |")

