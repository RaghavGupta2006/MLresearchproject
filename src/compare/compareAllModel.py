import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# -------------------------- 1. 全局样式配置（重点：调整标签行间距） --------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,          # 标签字体大小不变（两行显示足够清晰）
    'xtick.major.pad': 8,           # 增大刻度与标签的距离（避免标签贴柱子）
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
})

# -------------------------- 2. 数据准备（核心：标签用\n换行，拆成两行） --------------------------
# 2.1 传统ML模型：标签拆为“模型名\n(模态)”（如“XGBoost\n(Table)”）
traditional_ml = {
    'names': ['XGBoost\n(Table)', 'GBR\n(Table)'],  # 换行关键：\n分隔两行
    'test_r2': [0.8288, 0.8077],
    'test_rmse': [11.5455, 12.2355]
}

# 2.2 梯度提升模型：同样拆分为两行，保持格式统一
gb_models = {
    'single_modal_name': ['GB\n(Table Only)'],       # 单模态：“GB\n(Table Only)”
    'single_test_r2': [0.8494],
    'single_test_rmse': [11.2594],
    'multi_modal_names': [
        'GB\n(Table+Graph)',
        'GB\n(Table+Image)',
        'GB\n(Table+FP)'
    ],  # 多模态：“GB\n(Table+X)”
    'multi_test_r2': [0.9014, 0.8571, 0.7918],
    'multi_test_rmse': [9.1118, 10.9668, 13.2400]
}

# 2.3 合并数据（换行后的标签横向长度减少50%，彻底解决拥挤）
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
bar_width = 0.6  # 保持原柱子宽度（换行后横向空间足够，无需缩小）

# -------------------------- 3. 视觉区分配置（保持不变） --------------------------
colors = [
    '#808080', '#A9A9A9',  # 传统ML：深灰、浅灰
    '#2E86AB',              # 梯度提升单模态：蓝色
    '#A23B72', '#2E9F78', '#C73E1D'  # 梯度提升多模态：紫红、绿、红
]
hatches = ['/', '\\', '', '', '', '']

# -------------------------- 4. 绘制子图（重点：设置标签两行的行间距） --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # 保持原画布宽度（无需扩大）
fig.suptitle('Performance Comparison: Traditional ML vs Gradient Boosting (Single/Multi-modal)',
             fontsize=16, fontweight='bold', y=0.95)

# -------------------------- 子图1：测试集R²对比 --------------------------
bars_r2 = ax1.bar(
    x_index, all_test_r2, bar_width,
    color=colors, hatch=hatches,
    edgecolor='black', linewidth=0.8
)

ax1.set_title('(a) Test Set R² Score Comparison', fontweight='bold')
ax1.set_ylabel('R²')
ax1.set_xticks(x_index)
# 关键：用linespacing=1.2控制两行标签的间距（避免上下行太挤）
ax1.set_xticklabels(x_labels, rotation=0, ha='center', linespacing=1.2)
ax1.grid(True, axis='y')
ax1.set_ylim(0.75, 0.95)

# 数值标签（位置不变，不干扰两行标签）
for bar, r2 in zip(bars_r2, all_test_r2):
    ax1.text(
        bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
        f'{r2:.4f}', ha='center', va='bottom', fontsize=9
    )

# -------------------------- 子图2：测试集RMSE对比 --------------------------
bars_rmse = ax2.bar(
    x_index, all_test_rmse, bar_width,
    color=colors, hatch=hatches,
    edgecolor='black', linewidth=0.8
)

ax2.set_title('(b) Test Set RMSE Comparison', fontweight='bold')
ax2.set_ylabel('RMSE')  # 替换为你的实际单位
ax2.set_xticks(x_index)
# 统一两行标签格式：linespacing=1.2
ax2.set_xticklabels(x_labels, rotation=0, ha='center', linespacing=1.2)
ax2.grid(True, axis='y')
ax2.set_ylim(8, 14)

# 数值标签
for bar, rmse in zip(bars_rmse, all_test_rmse):
    ax2.text(
        bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
        f'{rmse:.4f}', ha='center', va='bottom', fontsize=9
    )

# -------------------------- 5. 自定义图例（保持不变） --------------------------
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

# -------------------------- 6. 布局调整（关键：增加底部间距，避免两行标签被遮挡） --------------------------
plt.subplots_adjust(
    bottom=0.22,  # 从0.18→0.22（两行标签纵向占用略多，预留足够空间）
    top=0.88,
    wspace=0.3
)

# -------------------------- 7. 保存图片 --------------------------
plt.savefig('traditional_ml_vs_gb_comparison_two_lines.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('traditional_ml_vs_gb_comparison_two_lines.pdf', bbox_inches='tight', facecolor='white')

plt.show()