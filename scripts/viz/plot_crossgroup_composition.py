"""
Visualize compositional steering: Russian × C features on Qwen3-4B.

Shows a 2×2 grid where x-axis = C direction, y-axis = Russian direction,
with representative model outputs at each combination. Demonstrates that
nearly orthogonal features (cos = 0.014) compose additively.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from textwrap import fill

# --- Curated snippets from fibonacci(n) { prompt, steered at 30° ---

snippets = {
    (0, 0): {  # baseline
        "label": "Baseline",
        "code": (
            "if(n <= 0) return 0;\n"
            "if(n == 1) return 1;\n"
            "return fibonacci(n-1)\n"
            "     + fibonacci(n-2);\n"
            "}\n\n"
            "function fibonacci(n) {\n"
            "  if(n <= 0) return 0;\n"
            "  if(n == 1) return 1;\n"
            "  ..."
        ),
        "desc": "JavaScript",
    },
    (1, 0): {  # +C only
        "label": "+C only",
        "code": (
            "if (n <= 1) return n;\n"
            "else return fibonacci(n-1)\n"
            "          + fibonacci(n-2);\n"
            "}\n\n"
            "// O(n) time\n"
            "// O(1) space\n"
            "// But for large numbers,\n"
            "// not acceptable. Use\n"
            "// memoization."
        ),
        "desc": "C-style + English comments",
    },
    (0, 1): {  # +RU only
        "label": "+RU only",
        "code": (
            "if (n === 0) return 0\n"
            "if (n === 1) return 1\n"
            "return fibonacci(n-1)\n"
            "     + fibonacci(n-2)\n"
            "}\n\n"
            "// Реализуй функцию,\n"
            "// которая по номеру n\n"
            "// возвращает n-е число\n"
            "// Фибоначчи"
        ),
        "desc": "JavaScript + Russian comments",
    },
    (1, 1): {  # +RU+C combined
        "label": "+RU + C",
        "code": (
            "if (n==0 || n==1)\n"
            "    return 1;\n"
            "else\n"
            "    return fibonacci(n-1)\n"
            "         + fibonacci(n-2);\n"
            "}\n\n"
            "// Реализация рекурсивного\n"
            "// Плюсы: простая реализация\n"
            "// Минус: медленная"
        ),
        "desc": "C-style + Russian comments",
    },
}

# --- Colors ---
bg_colors = {
    (0, 0): "#f0f0f0",   # neutral gray
    (1, 0): "#dce8f5",   # light blue (C)
    (0, 1): "#f5dce0",   # light red/pink (Russian)
    (1, 1): "#e0d5f0",   # light purple (combined)
}

border_colors = {
    (0, 0): "#999999",
    (1, 0): "#4a86c8",
    (0, 1): "#c84a5a",
    (1, 1): "#7b4ac8",
}

# --- Figure ---
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle(
    'Compositional Steering: Russian × C\n'
    'Prompt: "fibonacci(n) {"  ·  Qwen3-4B layer 14  ·  30° steering  ·  cos(RU, C) = 0.014',
    fontsize=13, fontweight='bold', y=0.98
)

for (cx, ry), snippet in snippets.items():
    ax = axes[1 - ry][cx]  # row 0 = top = +RU, col 0 = left = no C

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Background
    rect = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        facecolor=bg_colors[(cx, ry)],
        edgecolor=border_colors[(cx, ry)],
        linewidth=2.5,
    )
    ax.add_patch(rect)

    # Label (top-left)
    ax.text(0.06, 0.95, snippet["label"], fontsize=11, fontweight='bold',
            va='top', ha='left', color=border_colors[(cx, ry)])

    # Description tag (top-right)
    ax.text(0.94, 0.95, snippet["desc"], fontsize=8.5, fontstyle='italic',
            va='top', ha='right', color='#555555')

    # Code snippet (monospace, centered)
    ax.text(0.50, 0.48, snippet["code"], fontsize=8.5,
            fontfamily='monospace', va='center', ha='center',
            color='#1a1a1a',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', alpha=0.85))

# Axis labels
fig.text(0.5, 0.02, '← no C steering          C feature →', ha='center',
         fontsize=12, fontweight='bold', color='#4a86c8')
fig.text(0.01, 0.5, '← no RU steering          RU feature →', ha='center',
         fontsize=12, fontweight='bold', color='#c84a5a',
         rotation=90)

plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.07,
                    hspace=0.08, wspace=0.08)

save_path = "results/crossgroup_composition_figure.png"
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {save_path}")
plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved PDF too")
plt.show()
