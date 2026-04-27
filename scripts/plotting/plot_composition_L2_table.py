#!/usr/bin/env python3
"""Render the composition-table figure as a PNG/PDF (no LaTeX).

Four rows, one per steering condition, each with colour-coded completion.
Uses matplotlib text placement with explicit word wrapping.
"""
import os, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_PNG = 'writeup/figures/composition_table.png'
OUT_PDF = 'writeup/figures/composition_table.pdf'

# Colours
C_REG  = '#2a2a2a'
C_PT   = '#2166AC'
C_FEM  = '#C51B7D'
C_BOTH = '#7B3294'

# Each row: (label, subtitle, segments).
# segments is a flat list of (text, color) fragments — newlines inside text
# are preserved as paragraph breaks.
ROWS = [
    ('Baseline', 'no steering', [
        ('last year and have now appointed a new general manager.\n\n', C_REG),
        ('"The committee has decided to restructure the club to ensure that '
         'it is well placed to be able to compete at the highest level in '
         'the future," club chairman Mark Harris said.\n\n', C_REG),
        ('"The committee is confident that with the appointment of Michael, '
         'it will lead to the club playing at a very high standard."\n\n',
         C_REG),
        ('Harris said he was confident the', C_REG),
    ]),
    ('+ Portuguese', r'20°', [
        ('of the board of directors on Wednesday. He will succeed the '
         'outgoing president, who ', C_REG),
        ('termina seu mandado este ano', C_PT),
        ('.\n\n', C_REG),
        ('O encontro da Comissão Permanente da Organização da Unidade '
         'Africana (AOU) reuniu em Bangui, República Centro-Africana, de '
         '25 a 27 de Janeiro de 2023, para discutir o tema da segurança '
         'e desenvolvimento para', C_PT),
    ]),
    ('+ Female', r'20°', [
        (', who pledged to increase the amount of power devolved to local '
         'councillors.\n\n', C_REG),
        ('Councillor ', C_REG),
        ('Sue', C_FEM),
        (' Baxter is the first ', C_REG),
        ('woman', C_FEM),
        (' in 14 years to head the Conservative administration in North '
         'Yorkshire.\n\n', C_REG),
        ('She', C_FEM),
        (' was the first ', C_REG),
        ('female', C_FEM),
        (' to chair the authority\'s policy and resources committee and '
         'has held many executive and executive portfolio committee '
         'positions.\n\n', C_REG),
        ('"I feel very honoured to have been elected chairman of the '
         'council," ', C_REG),
        ('she', C_FEM),
        (' said', C_REG),
    ]),
    ('+ Portuguese\n+ Female', r'20° each', [
        ('and four vice-chairmen.\n\n', C_REG),
        ('The election for the chairman was between incumbent ', C_REG),
        ('chairwoman', C_FEM),
        (', ', C_REG),
        ('Maria Helena Braga', C_BOTH),
        (', ', C_PT),
        ('de 88 anos e a ', C_PT),
        ('doutra em medicina', C_BOTH),
        (', ', C_PT),
        ('Maria de Fátima', C_BOTH),
        (', ', C_PT),
        ('de 53 anos.\n\n', C_PT),
        ('No primeira votação, obteve 142 votos e a ', C_PT),
        ('doutra', C_BOTH),
        (' 137.\n\n', C_PT),
        ('No segunda votação, ', C_PT),
        ('Maria Braga', C_BOTH),
        (' obteve 13', C_PT),
    ]),
]

# ------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------

FIG_W_IN = 11.5
FONT_SIZE = 9.0
LINE_HEIGHT_PTS = FONT_SIZE * 1.5       # between wrapped lines
PARA_GAP_PTS = FONT_SIZE * 0.6           # extra gap between paragraphs
ROW_GAP_PTS = FONT_SIZE * 2.4            # between rows
LEFT_COL_W_IN = 2.25
TEXT_WRAP_CHARS = 115                    # approx chars per line at 9pt
TOP_MARGIN_IN = 0.55
BOTTOM_MARGIN_IN = 0.25


def fragment_to_tokens(text, color):
    """Split a fragment text into tokens preserving whitespace + newlines."""
    tokens = []
    buf = ''
    for ch in text:
        if ch == '\n':
            if buf:
                tokens.append((buf, color)); buf = ''
            tokens.append(('\n', color))
        elif ch == ' ':
            if buf:
                tokens.append((buf, color)); buf = ''
            tokens.append((' ', color))
        else:
            buf += ch
    if buf:
        tokens.append((buf, color))
    return tokens


def wrap_tokens(tokens, max_chars):
    """Return list of lines, each line is [(word, color), ...] with spaces
    rendered as explicit (' ', color) tokens. Paragraph breaks are lines
    containing a single ('', None, 'para') sentinel.
    """
    lines = []
    current = []
    cur_len = 0
    i = 0
    while i < len(tokens):
        tok, col = tokens[i]
        if tok == '\n':
            # Handle paragraph breaks: count consecutive \n's.
            nlines = 1
            while i + 1 < len(tokens) and tokens[i + 1][0] == '\n':
                nlines += 1
                i += 1
            # End current line.
            if current:
                lines.append(current); current = []; cur_len = 0
            if nlines >= 2:
                lines.append('PARA')   # paragraph gap
            i += 1
            continue
        if tok == ' ':
            if current:
                current.append((' ', col)); cur_len += 1
            i += 1
            continue
        # Regular word. Try to append.
        if cur_len + len(tok) > max_chars and current:
            # wrap
            # strip trailing spaces from line
            while current and current[-1][0] == ' ':
                current.pop()
            lines.append(current)
            current = []
            cur_len = 0
        current.append((tok, col))
        cur_len += len(tok)
        i += 1
    if current:
        lines.append(current)
    return lines


def build_all_lines():
    """Wrap every row into (row_idx, kind, data) entries."""
    entries = []
    for r_idx, (label, sub, segs) in enumerate(ROWS):
        # Build token stream
        tokens = []
        for txt, col in segs:
            tokens.extend(fragment_to_tokens(txt, col))
        lines = wrap_tokens(tokens, TEXT_WRAP_CHARS)
        entries.append((r_idx, label, sub, lines))
    return entries


def compute_height(entries):
    """Figure height based on line counts, in inches."""
    total_lines = 0
    para_count = 0
    for _, _, _, lines in entries:
        for ln in lines:
            if ln == 'PARA':
                para_count += 1
            else:
                total_lines += 1
    # pts to inches: 1 in = 72 pt
    lines_in = total_lines * LINE_HEIGHT_PTS / 72
    paras_in = para_count * PARA_GAP_PTS / 72
    rows_in = (len(entries) - 1) * ROW_GAP_PTS / 72
    header_in = 0.35                   # header row
    h = (TOP_MARGIN_IN + BOTTOM_MARGIN_IN + header_in + lines_in + paras_in +
         rows_in)
    return h


def main():
    entries = build_all_lines()
    fig_h = compute_height(entries)
    fig = plt.figure(figsize=(FIG_W_IN, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W_IN); ax.set_ylim(0, fig_h); ax.invert_yaxis()
    ax.axis('off')

    # Column geometry
    x_label = 0.4
    x_text = LEFT_COL_W_IN
    text_right = FIG_W_IN - 0.3

    # Header
    y = TOP_MARGIN_IN
    ax.text(x_label, y, 'Condition', fontsize=FONT_SIZE + 0.5,
            fontfamily='sans-serif', fontweight='bold', va='top')
    ax.text(x_text, y, 'Completion (prompt: "They elected a new chairman")',
            fontsize=FONT_SIZE + 0.5, fontfamily='sans-serif',
            fontweight='bold', va='top')
    y += 0.28
    # Header underline
    ax.plot([x_label - 0.05, text_right], [y, y], color='#333', lw=1.0)
    y += 0.12

    # Body rows
    for r_idx, label, sub, lines in entries:
        y_row_top = y
        # Label
        ax.text(x_label, y, label, fontsize=FONT_SIZE + 0.5,
                fontfamily='sans-serif', fontstyle='italic', va='top')
        n_label_lines = label.count('\n') + 1
        if sub:
            ax.text(x_label,
                    y + n_label_lines * LINE_HEIGHT_PTS / 72,
                    sub,
                    fontsize=FONT_SIZE - 1, color='#666',
                    fontfamily='sans-serif', va='top')
        # Body text: place each line as a sequence of text objects
        for ln in lines:
            if ln == 'PARA':
                y += PARA_GAP_PTS / 72
                continue
            x = x_text
            for tok, col in ln:
                t = ax.text(x, y, tok, fontsize=FONT_SIZE,
                            fontfamily='serif', color=col, va='top')
                # Measure width in axes coords
                fig.canvas.draw()
                bbox = t.get_window_extent()
                w_in = bbox.width / fig.dpi
                x += w_in
            y += LINE_HEIGHT_PTS / 72
        y += ROW_GAP_PTS / 72
        # Row separator
        ax.plot([x_label - 0.05, text_right], [y - 0.05, y - 0.05],
                color='#ddd', lw=0.5)

    # Legend / colour key
    y += 0.05
    legend_patches = [
        mpatches.Patch(color=C_PT,  label='Portuguese'),
        mpatches.Patch(color=C_FEM, label='feminine-gendered (English)'),
        mpatches.Patch(color=C_BOTH,label='feminine-gendered (Portuguese)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right',
              bbox_to_anchor=(1.0, 0.0), frameon=False, fontsize=FONT_SIZE,
              ncol=3)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {OUT_PNG}')
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
