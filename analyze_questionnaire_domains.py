import pandas as pd
import numpy as np

# قراءة البيانات
df = pd.read_excel('/home/ubuntu/upload/1CLSnewDatNOsum.xlsx')

# عرض جميع أسماء الأعمدة
print("=" * 80)
print("جميع المتغيرات في الاستبيان:")
print("=" * 80)
all_columns = df.columns.tolist()

# تحديد المتغيرات الديموغرافية (أول 8)
demographic_vars = ['Gender', 'Age', 'Fam_Hist', 'No_OfChild', 'Seq_ofChild', 
                    'Father_Emp', 'Mother_Emp', 'Blood_Relat']

# استخراج المتغيرات السلوكية (باستثناء الديموغرافية والتصنيف)
behavioral_vars = [col for col in all_columns 
                   if col not in demographic_vars and col != 'Class']

print(f"\nعدد المتغيرات السلوكية: {len(behavioral_vars)}")
print("\nالمتغيرات السلوكية:")
for i, var in enumerate(behavioral_vars, 1):
    print(f"{i}. {var}")

# تصنيف المتغيرات حسب المجالات
print("\n" + "=" * 80)
print("تصنيف المتغيرات حسب المجالات:")
print("=" * 80)

# 1. التفاعل الاجتماعي (Social Interaction)
social_interaction = [
    'Respond_name',           # الاستجابة للاسم
    'Eye_contact',            # التواصل البصري
    'Joint_attention',        # الانتباه المشترك
    'Social_smile',           # الابتسامة الاجتماعية
    'Imitation',              # التقليد
    'Pointing',               # الإشارة
    'Sharing_enjoyment',      # مشاركة المتعة
]

# 2. التواصل (Communication)
communication = [
    'Speech_clarity',         # وضوح الكلام
    'Vocabulary_size',        # حجم المفردات
    'Sentence_use',           # استخدام الجمل
    'Conversation',           # القدرة على المحادثة
    'Echolalia',              # الصدى اللفظي
]

# 3. السلوكيات النمطية والمتكررة (Stereotyped/Repetitive Behaviors)
repetitive_behaviors = [
    'Repetitive_behaviors',   # السلوكيات المتكررة
    'Repetitive_manipulation',# التلاعب المتكرر بالأشياء
    'Hand_flapping',          # رفرفة اليدين
    'Finger_move',            # حركة الأصابع
    'Spinning',               # الدوران
    'Rocking',                # التأرجح
    'walkONtiptoe',           # المشي على أطراف الأصابع
    'Blank_staring',          # التحديق الفارغ
]

# 4. الحساسية الحسية (Sensory Sensitivity)
sensory_sensitivity = [
    'Sensory_seeking',        # البحث الحسي
    'Sound_sensitivity',      # الحساسية للصوت
    'Light_sensitivity',      # الحساسية للضوء
    'Touch_sensitivity',      # الحساسية للمس
    'Smell_taste_sensitivity',# الحساسية للرائحة والطعم
]

# 5. التطور اللغوي (Language Development)
language_development = [
    'First_words',            # الكلمات الأولى
    'Language_regression',    # التراجع اللغوي
    'Speech_clarity',         # وضوح الكلام (يتداخل مع التواصل)
    'Vocabulary_size',        # حجم المفردات (يتداخل مع التواصل)
]

# طباعة التصنيف
domains = {
    'التفاعل الاجتماعي (Social Interaction)': social_interaction,
    'التواصل (Communication)': communication,
    'السلوكيات النمطية والمتكررة (Repetitive Behaviors)': repetitive_behaviors,
    'الحساسية الحسية (Sensory Sensitivity)': sensory_sensitivity,
    'التطور اللغوي (Language Development)': language_development,
}

for domain_name, variables in domains.items():
    print(f"\n{'='*80}")
    print(f"{domain_name}")
    print(f"{'='*80}")
    print(f"عدد المتغيرات: {len(variables)}\n")
    
    # التحقق من وجود المتغيرات في البيانات الفعلية
    existing_vars = [v for v in variables if v in behavioral_vars]
    missing_vars = [v for v in variables if v not in behavioral_vars]
    
    if existing_vars:
        print("المتغيرات الموجودة في البيانات:")
        for i, var in enumerate(existing_vars, 1):
            # حساب الارتباط مع Class
            if var in df.columns:
                correlation = df[var].corr(df['Class'])
                print(f"  {i}. {var} (ارتباط مع الشدة: {correlation:.3f})")
    
    if missing_vars:
        print("\nالمتغيرات غير الموجودة في البيانات:")
        for var in missing_vars:
            print(f"  - {var}")

# إنشاء جدول شامل
print("\n" + "=" * 80)
print("جدول شامل لجميع المتغيرات السلوكية مع تصنيفها:")
print("=" * 80)

# إنشاء DataFrame للتصنيف
classification_data = []

for var in behavioral_vars:
    # تحديد المجال
    domain = "غير مصنف"
    for domain_name, variables in domains.items():
        if var in variables:
            domain = domain_name.split('(')[0].strip()
            break
    
    # حساب الارتباط
    correlation = df[var].corr(df['Class'])
    
    # حساب الإحصائيات
    mean_val = df[var].mean()
    std_val = df[var].std()
    
    classification_data.append({
        'المتغير': var,
        'المجال': domain,
        'الارتباط مع الشدة': f"{correlation:.3f}",
        'المتوسط': f"{mean_val:.2f}",
        'الانحراف المعياري': f"{std_val:.2f}"
    })

classification_df = pd.DataFrame(classification_data)
classification_df = classification_df.sort_values('المجال')

print(classification_df.to_string(index=False))

# حفظ النتائج
classification_df.to_csv('/home/ubuntu/autism_analysis/questionnaire_domains_classification.csv', 
                         index=False, encoding='utf-8-sig')

# إحصائيات حسب المجال
print("\n" + "=" * 80)
print("إحصائيات حسب المجال:")
print("=" * 80)

domain_stats = classification_df.groupby('المجال').agg({
    'المتغير': 'count'
}).rename(columns={'المتغير': 'عدد المتغيرات'})

print(domain_stats)

print("\n" + "=" * 80)
print("تم حفظ التصنيف في: /home/ubuntu/autism_analysis/questionnaire_domains_classification.csv")
print("=" * 80)

