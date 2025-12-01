import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.patches as mpatches

# إعداد الخطوط العربية
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# قراءة البيانات
df = pd.read_excel('/home/ubuntu/upload/1CLSnewDatNOsum.xlsx')

# تصنيف المتغيرات (بأسماء الأعمدة الفعلية)
classification = {
    '1childName_response': 'التفاعل الاجتماعي',
    '2ChildEye_contact': 'التفاعل الاجتماعي',
    '3ChildObject_lining': 'السلوكيات النمطية',
    '4ChildSpeech_clarity': 'التواصل',
    '5ChildRequest_pointing ': 'التفاعل الاجتماعي',  # لاحظ المسافة في النهاية
    '6ChildJoint_attention': 'التفاعل الاجتماعي',
    '7ChildRestriced_interests': 'السلوكيات النمطية',
    '8ChildVocabulary_size': 'التطور اللغوي',
    '9ChildSymbolic_play': 'التفاعل الاجتماعي',
    '10ChildShared_attention': 'التفاعل الاجتماعي',
    '11ChildSensory_seeking': 'الحساسية الحسية',
    '12ChildHand_prompting': 'التفاعل الاجتماعي',
    '13Child_walkONtiptoe': 'السلوكيات النمطية',
    '14ChildAdaptability_Flexibility': 'السلوكيات النمطية',
    '15Child_empathy': 'التفاعل الاجتماعي',
    '16Child_Repetitive_behaviors': 'السلوكيات النمطية',
    '17ChildFirst_words': 'التطور اللغوي',
    '18ChildEchoin_voice': 'التواصل',
    '19ChildGesture_use': 'التواصل',
    '20ChildFinger_move': 'السلوكيات النمطية',
    '21ChildChecks_reactions': 'التفاعل الاجتماعي',
    '22ChildFocused_attention': 'التفاعل الاجتماعي',
    '23ChildRepetitive_manipulation': 'السلوكيات النمطية',
    '24ChildNoise_sensitivity': 'الحساسية الحسية',
    '25ChildBlank_staring': 'السلوكيات النمطية',
}

# إنشاء DataFrame للتحليل
results = []
for var, domain in classification.items():
    correlation = df[var].corr(df['Class'])
    results.append({
        'Variable': var,
        'Domain': domain,
        'Correlation': correlation
    })

results_df = pd.DataFrame(results)

# إحصائيات المجالات
domain_stats = results_df.groupby('Domain').agg({
    'Variable': 'count',
    'Correlation': 'mean'
}).rename(columns={'Variable': 'Count', 'Correlation': 'Avg_Correlation'})

# ترتيب المجالات
domain_order = [
    'التفاعل الاجتماعي',
    'التواصل',
    'السلوكيات النمطية',
    'الحساسية الحسية',
    'التطور اللغوي'
]

# الألوان
colors = {
    'التفاعل الاجتماعي': '#3498db',
    'التواصل': '#2ecc71',
    'السلوكيات النمطية': '#e74c3c',
    'الحساسية الحسية': '#f39c12',
    'التطور اللغوي': '#9b59b6'
}

# إنشاء الشكل الرئيسي
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. مخطط دائري - توزيع عدد الأسئلة
ax1 = fig.add_subplot(gs[0, 0])
counts = [domain_stats.loc[d, 'Count'] if d in domain_stats.index else 0 for d in domain_order]
colors_list = [colors[d] for d in domain_order]

wedges, texts, autotexts = ax1.pie(counts, labels=domain_order, autopct='%1.1f%%',
                                     colors=colors_list, startangle=90,
                                     textprops={'fontsize': 10, 'weight': 'bold'})
ax1.set_title('Distribution of Questions Across Domains\nتوزيع الأسئلة على المجالات', 
              fontsize=12, weight='bold', pad=20)

# 2. مخطط شريطي - عدد الأسئلة لكل مجال
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(range(len(domain_order)), counts, color=colors_list, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(domain_order)))
ax2.set_xticklabels([d.replace(' ', '\n') for d in domain_order], fontsize=9, rotation=0)
ax2.set_ylabel('Number of Questions\nعدد الأسئلة', fontsize=10, weight='bold')
ax2.set_title('Number of Questions per Domain\nعدد الأسئلة لكل مجال', 
              fontsize=12, weight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# إضافة القيم فوق الأعمدة
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(count)}',
             ha='center', va='bottom', fontsize=11, weight='bold')

# 3. مخطط شريطي أفقي - متوسط الارتباط
ax3 = fig.add_subplot(gs[0, 2])
avg_corrs = [domain_stats.loc[d, 'Avg_Correlation'] if d in domain_stats.index else 0 for d in domain_order]
bars = ax3.barh(range(len(domain_order)), avg_corrs, color=colors_list, edgecolor='black', linewidth=1.5)
ax3.set_yticks(range(len(domain_order)))
ax3.set_yticklabels(domain_order, fontsize=9)
ax3.set_xlabel('Average Correlation with Severity\nمتوسط الارتباط مع الشدة', fontsize=10, weight='bold')
ax3.set_title('Average Correlation per Domain\nمتوسط الارتباط لكل مجال', 
              fontsize=12, weight='bold', pad=20)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# إضافة القيم
for i, (bar, corr) in enumerate(zip(bars, avg_corrs)):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2.,
             f'{corr:.3f}',
             ha='left', va='center', fontsize=10, weight='bold', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.5))

# 4. Heatmap - الارتباط لكل متغير
ax4 = fig.add_subplot(gs[1, :])

# إعداد البيانات للـ heatmap
heatmap_data = []
for domain in domain_order:
    domain_vars = results_df[results_df['Domain'] == domain].sort_values('Correlation', ascending=False)
    correlations = domain_vars['Correlation'].values
    # ملء بقيم NaN للمحاذاة
    max_vars = max(counts)
    padded = list(correlations) + [np.nan] * (int(max_vars) - len(correlations))
    heatmap_data.append(padded)

heatmap_array = np.array(heatmap_data)

# رسم الـ heatmap
im = ax4.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.5)

# إضافة القيم في الخلايا
for i in range(len(domain_order)):
    for j in range(int(max(counts))):
        if not np.isnan(heatmap_array[i, j]):
            text = ax4.text(j, i, f'{heatmap_array[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8, weight='bold')

ax4.set_yticks(range(len(domain_order)))
ax4.set_yticklabels(domain_order, fontsize=10)
ax4.set_xticks(range(int(max(counts))))
ax4.set_xticklabels([f'Q{i+1}' for i in range(int(max(counts)))], fontsize=8)
ax4.set_xlabel('Question Rank (sorted by correlation)\nترتيب السؤال (مرتب حسب الارتباط)', 
               fontsize=10, weight='bold')
ax4.set_title('Correlation Heatmap for Each Variable\nخريطة حرارية للارتباط لكل متغير', 
              fontsize=12, weight='bold', pad=20)

# إضافة colorbar
cbar = plt.colorbar(im, ax=ax4, orientation='vertical', pad=0.02)
cbar.set_label('Correlation\nالارتباط', fontsize=10, weight='bold')

# 5. Box plot - توزيع الارتباطات
ax5 = fig.add_subplot(gs[2, 0])

box_data = [results_df[results_df['Domain'] == d]['Correlation'].values for d in domain_order]
bp = ax5.boxplot(box_data, labels=[d.replace(' ', '\n') for d in domain_order],
                 patch_artist=True, showmeans=True)

# تلوين الصناديق
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax5.set_ylabel('Correlation with Severity\nالارتباط مع الشدة', fontsize=10, weight='bold')
ax5.set_title('Distribution of Correlations per Domain\nتوزيع الارتباطات لكل مجال', 
              fontsize=12, weight='bold', pad=20)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
ax5.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

# 6. مخطط شريطي مكدس - النسب المئوية
ax6 = fig.add_subplot(gs[2, 1])

total = sum(counts)
percentages = [(c/total)*100 for c in counts]

bottom = 0
for i, (pct, color, domain) in enumerate(zip(percentages, colors_list, domain_order)):
    ax6.bar(0, pct, bottom=bottom, color=color, edgecolor='black', linewidth=1.5, width=0.5)
    # إضافة النص
    if pct > 5:  # فقط للنسب الكبيرة
        ax6.text(0, bottom + pct/2, f'{domain}\n{pct:.1f}%', 
                ha='center', va='center', fontsize=9, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
    bottom += pct

ax6.set_xlim(-0.5, 0.5)
ax6.set_ylim(0, 100)
ax6.set_ylabel('Percentage (%)\nالنسبة المئوية', fontsize=10, weight='bold')
ax6.set_title('Percentage Distribution\nالتوزيع النسبي', fontsize=12, weight='bold', pad=20)
ax6.set_xticks([])
ax6.grid(axis='y', alpha=0.3, linestyle='--')

# 7. Top 10 متغيرات
ax7 = fig.add_subplot(gs[2, 2])

top10 = results_df.nlargest(10, 'Correlation')
top10_colors = [colors[d] for d in top10['Domain']]

bars = ax7.barh(range(10), top10['Correlation'].values, color=top10_colors, 
                edgecolor='black', linewidth=1.5)
ax7.set_yticks(range(10))
var_names = [v.replace('Child', '').replace('child', '').replace('\n', '')[:20] for v in top10['Variable']]
ax7.set_yticklabels(var_names, fontsize=8)
ax7.set_xlabel('Correlation\nالارتباط', fontsize=10, weight='bold')
ax7.set_title('Top 10 Variables by Correlation\nأفضل 10 متغيرات حسب الارتباط', 
              fontsize=12, weight='bold', pad=20)
ax7.grid(axis='x', alpha=0.3, linestyle='--')

# إضافة القيم
for i, (bar, corr) in enumerate(zip(bars, top10['Correlation'].values)):
    width = bar.get_width()
    ax7.text(width, bar.get_y() + bar.get_height()/2.,
             f' {corr:.3f}',
             ha='left', va='center', fontsize=8, weight='bold')

# العنوان الرئيسي
fig.suptitle('Comprehensive Analysis of Behavioral Domains in Autism Assessment\n' +
             'تحليل شامل للمجالات السلوكية في تقييم التوحد',
             fontsize=16, weight='bold', y=0.98)

plt.savefig('/home/ubuntu/autism_analysis/figures/domains_comprehensive_analysis.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ تم حفظ: domains_comprehensive_analysis.png")

# ============================================================================
# رسم إضافي: مخطط تفاعلي للمجالات
# ============================================================================

fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, domain in enumerate(domain_order):
    ax = axes[idx]
    
    domain_vars = results_df[results_df['Domain'] == domain].sort_values('Correlation', ascending=False)
    
    if len(domain_vars) > 0:
        var_names = [v.replace('Child', '').replace('child', '').replace('\n', '')[:25] 
                     for v in domain_vars['Variable']]
        correlations = domain_vars['Correlation'].values
        
        bars = ax.barh(range(len(var_names)), correlations, 
                       color=colors[domain], edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_yticks(range(len(var_names)))
        ax.set_yticklabels(var_names, fontsize=9)
        ax.set_xlabel('Correlation\nالارتباط', fontsize=10, weight='bold')
        ax.set_title(f'{domain}\n({len(var_names)} questions)', 
                     fontsize=11, weight='bold', pad=10,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[domain], 
                              edgecolor='black', alpha=0.3))
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # إضافة القيم
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {corr:.2f}',
                   ha='left' if width > 0 else 'right', 
                   va='center', fontsize=8, weight='bold')

# إخفاء المحور السادس
axes[5].axis('off')

# إضافة ملخص في المحور السادس
ax_summary = axes[5]
ax_summary.text(0.5, 0.9, 'Summary Statistics\nالإحصائيات الموجزة', 
                ha='center', va='top', fontsize=14, weight='bold',
                transform=ax_summary.transAxes)

summary_text = f"""
Total Questions: {len(results_df)}
إجمالي الأسئلة: {len(results_df)}

Domain Distribution:
توزيع المجالات:
"""

y_pos = 0.75
for domain in domain_order:
    count = len(results_df[results_df['Domain'] == domain])
    avg_corr = results_df[results_df['Domain'] == domain]['Correlation'].mean()
    summary_text = f"{domain}: {count} ({count/len(results_df)*100:.1f}%)\nAvg Corr: {avg_corr:.3f}"
    
    ax_summary.text(0.1, y_pos, summary_text,
                   ha='left', va='top', fontsize=9,
                   transform=ax_summary.transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[domain], 
                            edgecolor='black', alpha=0.3))
    y_pos -= 0.15

fig2.suptitle('Detailed Analysis by Domain\nتحليل تفصيلي حسب المجال',
              fontsize=16, weight='bold')
plt.tight_layout()

plt.savefig('/home/ubuntu/autism_analysis/figures/domains_detailed_analysis.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ تم حفظ: domains_detailed_analysis.png")

# ============================================================================
# رسم ثالث: مقارنة شاملة
# ============================================================================

fig3, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. مقارنة العدد والارتباط
ax1 = axes[0]
x = np.arange(len(domain_order))
width = 0.35

counts_normalized = [c/max(counts) for c in counts]
corrs_normalized = [c/max(avg_corrs) if max(avg_corrs) > 0 else 0 for c in avg_corrs]

bars1 = ax1.bar(x - width/2, counts_normalized, width, label='Normalized Count\nالعدد المعياري',
                color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, corrs_normalized, width, label='Normalized Avg Corr\nالارتباط المعياري',
                color='coral', edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Normalized Value\nالقيمة المعيارية', fontsize=11, weight='bold')
ax1.set_title('Comparison: Count vs Correlation\nمقارنة: العدد مقابل الارتباط', 
              fontsize=12, weight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([d.replace(' ', '\n') for d in domain_order], fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 2. Radar chart
ax2 = axes[1]
ax2.remove()
ax2 = fig3.add_subplot(132, projection='polar')

angles = np.linspace(0, 2 * np.pi, len(domain_order), endpoint=False).tolist()
counts_radar = counts + [counts[0]]
angles += angles[:1]

ax2.plot(angles, counts_radar, 'o-', linewidth=2, color='steelblue', label='Count')
ax2.fill(angles, counts_radar, alpha=0.25, color='steelblue')
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels([d.replace(' ', '\n') for d in domain_order], fontsize=8)
ax2.set_title('Radar Chart: Questions per Domain\nمخطط راداري: الأسئلة لكل مجال', 
              fontsize=12, weight='bold', pad=20)
ax2.grid(True)

# 3. Importance Score (Count × Avg Correlation)
ax3 = axes[2]

importance_scores = [c * ac for c, ac in zip(counts, avg_corrs)]
bars = ax3.bar(range(len(domain_order)), importance_scores, 
               color=colors_list, edgecolor='black', linewidth=1.5)

ax3.set_xticks(range(len(domain_order)))
ax3.set_xticklabels([d.replace(' ', '\n') for d in domain_order], fontsize=9)
ax3.set_ylabel('Importance Score\n(Count × Avg Correlation)\nدرجة الأهمية', 
               fontsize=10, weight='bold')
ax3.set_title('Domain Importance Score\nدرجة أهمية المجال', 
              fontsize=12, weight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# إضافة القيم
for i, (bar, score) in enumerate(zip(bars, importance_scores)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.2f}',
             ha='center', va='bottom', fontsize=10, weight='bold')

fig3.suptitle('Comprehensive Domain Comparison\nمقارنة شاملة للمجالات',
              fontsize=16, weight='bold')
plt.tight_layout()

plt.savefig('/home/ubuntu/autism_analysis/figures/domains_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ تم حفظ: domains_comparison.png")

print("\n" + "="*80)
print("تم إنشاء 3 تصورات بيانية شاملة بنجاح!")
print("="*80)
print("1. domains_comprehensive_analysis.png - تحليل شامل (7 رسوم)")
print("2. domains_detailed_analysis.png - تحليل تفصيلي لكل مجال")
print("3. domains_comparison.png - مقارنة شاملة")
print("="*80)

