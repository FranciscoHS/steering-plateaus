"""
Compositional steering: Portuguese × Gender on Gemma-2-9B.

Highlighted text examples where Portuguese text is blue, female-gendered
English words are pink, and Portuguese female-gendered words are purple
(the composition of both features).

Uses the prompt "They elected a new chairman" from the 20° experiment.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# === Colors ===
C_REG = '#333333'       # regular English text
C_PT = '#2166ac'        # Portuguese text (blue)
C_FEM = '#c51b7d'       # female-gendered English (pink/magenta)
C_BOTH = '#7b3294'      # both features: Portuguese + female (purple)
C_MUTED = '#777777'

BG = {
    'base': '#f5f5f5',
    'pt':   '#e8f0fa',
    'fem':  '#fce4ec',
    'both': '#f0e6f6',
}
BORDER = {
    'base': '#999999',
    'pt':   '#2166ac',
    'fem':  '#c51b7d',
    'both': '#7b3294',
}

FONTSIZE = 8.5
LINE_SPACING = 1.6


def measure_char_width(fig, ax, fontsize):
    """Measure exact monospace character width in axes coordinates.

    Creates visible text objects (invisible ones return wrong extents),
    measures, then removes them.
    """
    t1 = ax.text(0, 0, 'M' * 20, fontsize=fontsize, fontfamily='monospace',
                 transform=ax.transAxes)
    t2 = ax.text(0, 0, 'M' * 40, fontsize=fontsize, fontfamily='monospace',
                 transform=ax.transAxes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb1 = t1.get_window_extent(renderer)
    bb2 = t2.get_window_extent(renderer)
    inv = ax.transAxes.inverted()
    w1 = inv.transform((bb1.x1, 0))[0] - inv.transform((bb1.x0, 0))[0]
    w2 = inv.transform((bb2.x1, 0))[0] - inv.transform((bb2.x0, 0))[0]
    char_w = (w2 - w1) / 20  # differential eliminates any fixed offset
    t1.remove()
    t2.remove()
    return char_w


def draw_cell_text(ax, x0, y0, text_lines, base_color, highlights, char_w):
    """Render text with colored highlights.

    Each segment is positioned at x0 + col * char_w (absolute column position),
    avoiding cumulative width errors. Lines are split at highlight boundaries
    so each segment has a single color.
    """
    bbox_ax = ax.get_position()
    ax_h_pts = bbox_ax.height * ax.figure.get_figheight() * 72
    line_h = FONTSIZE * LINE_SPACING / ax_h_pts

    for i, line in enumerate(text_lines):
        y = y0 - i * line_h
        line_hls = highlights.get(i, [])

        if not line_hls:
            # No highlights on this line — render as single text object
            ax.text(x0, y, line, fontsize=FONTSIZE, fontfamily='monospace',
                    color=base_color, transform=ax.transAxes, va='top', ha='left')
            continue

        # Build segments: (col, text, color) — split at highlight boundaries
        segments = []
        pos = 0
        for col, hl_text, hl_color in sorted(line_hls, key=lambda h: h[0]):
            if col > pos:
                segments.append((pos, line[pos:col], base_color))
            segments.append((col, hl_text, hl_color))
            pos = col + len(hl_text)
        if pos < len(line):
            segments.append((pos, line[pos:], base_color))

        # Render each segment at its absolute column position
        for col, text, color in segments:
            x = x0 + col * char_w
            ax.text(x, y, text, fontsize=FONTSIZE, fontfamily='monospace',
                    color=color, transform=ax.transAxes, va='top', ha='left')


# === Cell data ===
# Prompt: "They elected a new chairman"
# Completions from Portuguese × Gender at 20°, Gemma-2-9B layer 20

cells = {
    (0, 0): {
        'label': 'Baseline',
        'desc': 'English · male',
        'style': 'base',
        'base_color': C_REG,
        'text': [
            "the former prime minister and",
            "president of Bosnia-Herzegovina",
            "Sefer Halilovic, who replaces",
            "the former chairman Mirko",
            "Pejanovic. The other members",
            "of the new executive committee",
            "are...",
        ],
        'highlights': {},
    },
    (0, 1): {
        'label': '+Portuguese (20°)',
        'desc': 'Portuguese · male',
        'style': 'pt',
        'base_color': C_PT,
        'text': [
            "João Neves, que substituiu a",
            "José Manuel Fonseca, para o",
            "cargo, que também exerceu por",
            "seis anos. A nova presidente,",
            "que trabalhou como advogado,",
            "teve um longo percurso na",
            "economia e no sector privado.",
        ],
        'highlights': {},
    },
    (1, 0): {
        'label': '+Female (20°)',
        'desc': 'English · female',
        'style': 'fem',
        'base_color': C_REG,
        'text': [
            "a new vice chair and a new",
            "secretary but they also passed",
            "a resolution to rename the",
            'club the "New" Women\'s',
            "Institute. ...said",
            "Margaret McWilliams.",
            "McWilliams and her sister",
        ],
        'highlights': {
            3: [(15, "Women's", C_FEM)],
            5: [(0, "Margaret McWilliams", C_FEM)],
            6: [(15, "her sister", C_FEM)],
        },
    },
    (1, 1): {
        'label': '+PT +Female (20°)',
        'desc': 'Portuguese · female',
        'style': 'both',
        'base_color': C_PT,
        'text': [
            "a jovem professora da",
            "Universidade do Minho, a",
            "Dra. Maria João Mendes,",
            "substituindo a professora",
            "catedrária da Universidade de",
            "Coimbra, a Dra. Maria José",
            "Pereira. A nova presidente...",
        ],
        'highlights': {
            0: [(2, "jovem professora", C_BOTH)],
            2: [(0, "Dra. Maria João Mendes", C_BOTH)],
            3: [(15, "professora", C_BOTH)],
            4: [(0, "catedrária", C_BOTH)],
            5: [(11, "Dra. Maria José", C_BOTH)],
            6: [(0, "Pereira", C_BOTH), (11, "nova presidente", C_BOTH)],
        },
    },
}


# === Build figure ===
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.07,
                    hspace=0.08, wspace=0.08)

fig.suptitle(
    'Compositional Steering: Portuguese × Gender\n'
    'Prompt: "They elected a new chairman"  ·  '
    'Gemma-2-9B layer 20  ·  20°  ·  cos(PT, ♀) = −0.021',
    fontsize=13, fontweight='bold', y=0.98,
)

# Measure char width once (all axes same size in uniform grid)
char_w = measure_char_width(fig, axes[0][0], FONTSIZE)

for (cx, ry), cell in cells.items():
    ax = axes[1 - ry][cx]  # row 0 = top = +PT

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Rounded background
    rect = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        facecolor=BG[cell['style']],
        edgecolor=BORDER[cell['style']],
        linewidth=2.5,
    )
    ax.add_patch(rect)

    # Label (top-left)
    ax.text(0.06, 0.95, cell['label'], fontsize=11, fontweight='bold',
            va='top', ha='left', color=BORDER[cell['style']])

    # Description (top-right)
    ax.text(0.94, 0.95, cell['desc'], fontsize=9, fontstyle='italic',
            va='top', ha='right', color=C_MUTED)

    # Text body with highlights
    draw_cell_text(ax, 0.08, 0.80, cell['text'], cell['base_color'],
                   cell['highlights'], char_w)


# Axis labels
fig.text(0.5, 0.04,
         '← no gender steering              ♀ female direction →',
         ha='center', fontsize=12, fontweight='bold', color=C_FEM)
fig.text(0.01, 0.20,
         '← no language steering              PT direction →',
         ha='center', fontsize=12, fontweight='bold', color=C_PT, rotation=90)

# Legend
legend_handles = [
    mpatches.Patch(color=C_PT,   label='Portuguese text'),
    mpatches.Patch(color=C_FEM,  label='Female-gendered (English)'),
    mpatches.Patch(color=C_BOTH, label='Female-gendered (Portuguese)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))

save_path = "results/mandarin_haskell/gemma-2-9b/composition_pt_gender_highlight.png"
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {save_path}")
plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved PDF too")
