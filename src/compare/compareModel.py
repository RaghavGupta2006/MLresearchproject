import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 1.   --------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.fontsize': 11,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
})

# -------------------------- 2.   --------------------------
models = ['Table Only', 'Table+Graph', 'Table+Image', 'Table+FP']  # Note: processed parameter
rmse_unit = ' (kPa)'  # Note: processed parameter

# Train Set
train_r2 = [0.9460, 0.9635, 0.9786, 0.9356]
train_rmse = [6.2717, 5.1581, 3.9490, 6.8470]

# Test Set
test_r2 = [0.8494, 0.9014, 0.8571, 0.7918]
test_rmse = [11.2594, 9.1118, 10.9668, 13.2400]

# Table Only
perf_impr_r2 = [test_r2[i] - test_r2[0] for i in range(len(test_r2))]
perf_impr_rmse = [test_rmse[0] - test_rmse[i] for i in range(len(test_rmse))]

# -------------------------- 3.   --------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='col')
fig.suptitle('Performance Comparison of Different Multimodal Models',
             fontsize=16, fontweight='bold', y=0.98)

bar_width = 0.35
x_index = np.arange(len(models))
colors = {
    'train': '# 2E86AB',    #  Train Set
    'test': '# A23B72'      #  Test Set
}
impr_colors = ['# 2E9F78' if x > 0 else '#C73E1D' for x in perf_impr_r2]  #

# Note: processed parameter
star_kwargs = dict(ha='center', va='bottom', fontsize=14, color='black')

# -------------------------- (a) R² Comparison Test Set *  --------------------------
ax1 = axes[0, 0]
bars1_train = ax1.bar(x_index - bar_width/2, train_r2, bar_width,
                     label='Training', color=colors['train'], alpha=0.8,
                     edgecolor='black', linewidth=0.8)
bars1_test = ax1.bar(x_index + bar_width/2, test_r2, bar_width,
                    label='Testing', color=colors['test'], alpha=0.8,
                    edgecolor='black', linewidth=[0.8 if i != 1 else 1.8 for i in range(len(models))])

ax1.set_title('(a) R² Comparison', fontweight='bold')
ax1.set_ylabel('R²')
ax1.set_xticks(x_index)
ax1.set_xticklabels(models, rotation=0, ha='center')  # Note: processed parameter
ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax1.grid(True, axis='y')
ax1.set_ylim(0.75, 1.03)

for i, (bar_train, bar_test) in enumerate(zip(bars1_train, bars1_test)):
    ax1.text(bar_train.get_x() + bar_train.get_width()/2., bar_train.get_height() + 0.005,
             f'{bar_train.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    fontweight = 'bold' if i == 1 else 'normal'
    ax1.text(bar_test.get_x() + bar_test.get_width()/2., bar_test.get_height() + 0.005,
             f'{bar_test.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight=fontweight)

# *
test_bar_x = x_index[1] + bar_width/2
test_bar_height = test_r2[1]
ax1.text(test_bar_x, test_bar_height + 0.018, '*', **star_kwargs)

# -------------------------- (b) RMSE Comparison Test Set *  --------------------------
ax2 = axes[0, 1]
bars2_train = ax2.bar(x_index - bar_width/2, train_rmse, bar_width,
                     label='Training', color=colors['train'], alpha=0.8,
                     edgecolor='black', linewidth=0.8)
bars2_test = ax2.bar(x_index + bar_width/2, test_rmse, bar_width,
                    label='Testing', color=colors['test'], alpha=0.8,
                    edgecolor='black', linewidth=[0.8 if i != 1 else 1.8 for i in range(len(models))])

ax2.set_title('(b) RMSE Comparison', fontweight='bold')
ax2.set_ylabel(f'RMSE{rmse_unit}')
ax2.set_xticks(x_index)
ax2.set_xticklabels(models, rotation=0, ha='center')  # Note: processed parameter
ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax2.grid(True, axis='y')
ax2.set_ylim(3, 14.5)

for i, (bar_train, bar_test) in enumerate(zip(bars2_train, bars2_test)):
    ax2.text(bar_train.get_x() + bar_train.get_width()/2., bar_train.get_height() + 0.2,
             f'{bar_train.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    fontweight = 'bold' if i == 1 else 'normal'
    ax2.text(bar_test.get_x() + bar_test.get_width()/2., bar_test.get_height() + 0.2,
             f'{bar_test.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight=fontweight)

# *
test_bar_x = x_index[1] + bar_width/2
test_bar_height = test_rmse[1]
ax2.text(test_bar_x, test_bar_height + 0.6, '*', **star_kwargs)

# -------------------------- (c) ΔR² * x  --------------------------
ax3 = axes[1, 0]
bars3 = ax3.bar(x_index, perf_impr_r2, bar_width*1.2,
               color=impr_colors, alpha=0.8, edgecolor='black', linewidth=0.8)

ax3.set_title('(c) ΔR² (vs Table Only)', fontweight='bold')
ax3.set_ylabel('ΔR²')
# ax3.set_xlabel('Model Type')  #   x
ax3.set_xticks(x_index)
ax3.set_xticklabels(models, rotation=0, ha='center')  # Note: processed parameter
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.9, linewidth=1.2)
ax3.grid(True, axis='y')
ax3.set_ylim(-0.07, 0.06)

for bar in bars3:
    h = bar.get_height()
    va = 'bottom' if h >= 0 else 'top'
    y_offset = 0.003 if h >= 0 else -0.005
    ax3.text(bar.get_x() + bar.get_width()/2., h + y_offset,
             f'{h:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

# -------------------------- (d) ΔRMSE * x  --------------------------
ax4 = axes[1, 1]
impr_colors_rmse = ['#2E9F78' if x > 0 else '#C73E1D' for x in perf_impr_rmse]
bars4 = ax4.bar(x_index, perf_impr_rmse, bar_width*1.2,
               color=impr_colors_rmse, alpha=0.8, edgecolor='black', linewidth=0.8)

ax4.set_title('(d) ΔRMSE (vs Table Only)', fontweight='bold')
ax4.set_ylabel(f'ΔRMSE{rmse_unit}')
# ax4.set_xlabel('Model Type')  #   x
ax4.set_xticks(x_index)
ax4.set_xticklabels(models, rotation=0, ha='center')  # Note: processed parameter
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.9, linewidth=1.2)
ax4.grid(True, axis='y')
ax4.set_ylim(-2.5, 2.5)

for bar in bars4:
    h = bar.get_height()
    va = 'bottom' if h >= 0 else 'top'
    y_offset = 0.08 if h >= 0 else -0.08
    ax4.text(bar.get_x() + bar.get_width()/2., h + y_offset,
             f'{h:+.2f}', ha='center', va=va, fontsize=9, fontweight='bold')

# --------------------------   --------------------------
plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.35)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Note: processed parameter
plt.savefig('model_comparison_final.png', dpi=300, bbox_inches='tight')
plt.savefig('model_comparison_final.pdf', bbox_inches='tight')

plt.show()