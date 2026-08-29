import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# -------------------------- 1.   --------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,          # Note: processed parameter
    'xtick.major.pad': 8,           # Note: processed parameter
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
})

# -------------------------- 2.  \n  --------------------------
# 2.1  ML “ \n( )” “XGBoost\n(Table)”
traditional_ml = {
    'names': ['XGBoost\n(Table)', 'GBR\n(Table)'],  # \n
    'test_r2': [0.8288, 0.8077],
    'test_rmse': [11.5455, 12.2355]
}

# 2.2
gb_models = {
    'single_modal_name': ['GB\n(Table Only)'],       # “GB\n(Table Only)”
    'single_test_r2': [0.8494],
    'single_test_rmse': [11.2594],
    'multi_modal_names': [
        'GB\n(Table+Graph)',
        'GB\n(Table+Image)',
        'GB\n(Table+FP)'
    ],  # “GB\n(Table+X)”
    'multi_test_r2': [0.9014, 0.8571, 0.7918],
    'multi_test_rmse': [9.1118, 10.9668, 13.2400]
}

# 2.3  50%
x_labels = (traditional_ml['names'] +
            gb_models['single_modal_name'] +
            gb_models['multi_modal_names'])
all_test_r2 = (traditional_ml['test_r2'] +
               gb_models['single_test_r2'] +
               gb_models['multi_test_r2'])
all_test_rmse = (traditional_ml['test_rmse'] +
                 gb_models['single_test_rmse'] +
                 gb_models['multi_test_rmse'])

x_index = np.arange(len(x_labels))
bar_width = 0.6  # Note: processed parameter

# -------------------------- 3.   --------------------------
colors = [
    '# 808080', '#A9A9A9',  #  ML 、
    '# 2E86AB',              #
    '# A23B72', '#2E9F78', '#C73E1D'  #  、 、
]
hatches = ['/', '\\', '', '', '', '']

# -------------------------- 4.   --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # Note: processed parameter
fig.suptitle('Performance Comparison: Traditional ML vs Gradient Boosting (Single/Multi-modal)',
             fontsize=16, fontweight='bold', y=0.95)

# --------------------------  1 Test SetR²  --------------------------
bars_r2 = ax1.bar(
    x_index, all_test_r2, bar_width,
    color=colors, hatch=hatches,
    edgecolor='black', linewidth=0.8
)

ax1.set_title('(a) Test Set R² Score Comparison', fontweight='bold')
ax1.set_ylabel('R²')
ax1.set_xticks(x_index)
# linespacing=1.2
ax1.set_xticklabels(x_labels, rotation=0, ha='center', linespacing=1.2)
ax1.grid(True, axis='y')
ax1.set_ylim(0.75, 0.95)

# Value
for bar, r2 in zip(bars_r2, all_test_r2):
    ax1.text(
        bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
        f'{r2:.4f}', ha='center', va='bottom', fontsize=9
    )

# --------------------------  2 Test SetRMSE  --------------------------
bars_rmse = ax2.bar(
    x_index, all_test_rmse, bar_width,
    color=colors, hatch=hatches,
    edgecolor='black', linewidth=0.8
)

ax2.set_title('(b) Test Set RMSE Comparison', fontweight='bold')
ax2.set_ylabel('RMSE')  # Note: processed parameter
ax2.set_xticks(x_index)
# linespacing=1.2
ax2.set_xticklabels(x_labels, rotation=0, ha='center', linespacing=1.2)
ax2.grid(True, axis='y')
ax2.set_ylim(8, 14)

# Value
for bar, rmse in zip(bars_rmse, all_test_rmse):
    ax2.text(
        bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
        f'{rmse:.4f}', ha='center', va='bottom', fontsize=9
    )

# -------------------------- 5.   --------------------------
legend_elements = [
    Patch(facecolor='#808080', hatch='/', edgecolor='black', label='Traditional ML: XGBoost (Table)'),
    Patch(facecolor='#A9A9A9', hatch='\\', edgecolor='black', label='Traditional ML: GBR (Table)'),
    Patch(facecolor='#2E86AB', edgecolor='black', label='Gradient Boosting: Single-modal (Table)'),
    Patch(facecolor='#A23B72', edgecolor='black', label='Gradient Boosting: Multi-modal (Table+Graph)'),
    Patch(facecolor='#2E9F78', edgecolor='black', label='Gradient Boosting: Multi-modal (Table+Image)'),
    Patch(facecolor='#C73E1D', edgecolor='black', label='Gradient Boosting: Multi-modal (Table+FP)')
]

fig.legend(
    handles=legend_elements, loc='lower center',
    bbox_to_anchor=(0.5, 0.02), ncol=3,
    columnspacing=1.2, handlelength=1.5
)

# -------------------------- 6.   --------------------------
plt.subplots_adjust(
    bottom=0.22,  # 0.18→0.22
    top=0.88,
    wspace=0.3
)

# -------------------------- 7.   --------------------------
plt.savefig('traditional_ml_vs_gb_comparison_two_lines.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('traditional_ml_vs_gb_comparison_two_lines.pdf', bbox_inches='tight', facecolor='white')

plt.show()