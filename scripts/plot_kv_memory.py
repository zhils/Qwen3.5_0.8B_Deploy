import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

seq_lens = [64, 128, 256, 512, 1024, 2048]

num_layers = 24
num_kv_heads = 2
head_dim = 256
max_seq_len = 2048
bytes_per_token = num_layers * num_kv_heads * head_dim * 4 * 2

fp32_nonpaged = [max_seq_len * bytes_per_token / (1024**2) for _ in seq_lens]

paged_data = {
    64: 6.0, 128: 12.0, 256: 24.0, 512: 48.0,
    1024: 96.0, 2048: 192.0
}
paged = [paged_data[seq] for seq in seq_lens]

int8_paged = [p / 4 for p in paged]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(seq_lens))
width = 0.25

bars1 = ax.bar(x - width, fp32_nonpaged, width, label='FP32 Non-Paged (Pre-allocated)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x, paged, width, label='FP32 Paged (On-demand)', color='#3498db', alpha=0.8)
bars3 = ax.bar(x + width, int8_paged, width, label='INT8 + Paged', color='#2ecc71', alpha=0.8)

ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
ax.set_ylabel('KV Cache Memory (MB)', fontsize=12)
ax.set_title('KV Cache Memory Usage Comparison\n(Qwen3.5-0.8B, 24 layers, 2 KV heads, 256 head_dim)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(seq_lens)
ax.legend(loc='upper left', fontsize=10)

ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

textstr = '\n'.join([
    'Key Findings:',
    '',
    'FP32 Non-Paged: Pre-allocates max_seq_len',
    f'  Always uses 192 MB (max_seq=2048)',
    '',
    'FP32 Paged: On-demand allocation',
    '  seq=64: 6 MB, seq=2048: 192 MB',
    '',
    'INT8 + Paged: Best of both',
    '  seq=2048: only 48 MB (4x reduction)',
    '',
    'Memory savings at seq=64:',
    '  Non-Paged: 192 MB',
    '  Paged: 6 MB (32x savings!)'
])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.98, 0.55, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('docs/kv_memory_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('kv_memory_comparison.png', dpi=150, bbox_inches='tight')
print("Chart saved to docs/kv_memory_comparison.png")

print("\n=== KV Memory Comparison Data ===")
print(f"{'seq_len':>8} | {'FP32 Non-Paged':>17} | {'FP32 Paged':>12} | {'INT8+Paged':>12} | {'Paged Savings':>14} | {'INT8 Savings':>13}")
print("-" * 95)
for i, seq in enumerate(seq_lens):
    paged_savings = fp32_nonpaged[i] / paged[i]
    int8_savings = fp32_nonpaged[i] / int8_paged[i]
    print(f"{seq:>8} | {fp32_nonpaged[i]:>17.1f} MB | {paged[i]:>12.1f} MB | {int8_paged[i]:>12.1f} MB | {paged_savings:>14.1f}x | {int8_savings:>13.1f}x")

print("\n=== Conclusion ===")
print("1. FP32 Non-Paged: Pre-allocates memory for max_seq_len, wastes memory for short sequences")
print("2. FP32 Paged: On-demand allocation, saves up to 32x memory for short sequences (seq=64)")
print("3. INT8 + Paged: Combines both benefits, achieves up to 128x memory reduction (seq=64)")
print("   At seq=2048: 192MB -> 48MB (4x reduction), enabling longer context on limited VRAM")
