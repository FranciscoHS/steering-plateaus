#!/usr/bin/env python3
"""Composition figure: Formal × Portuguese on "she's literally the worst, omg".

Mimics the rounded-box, monospace-with-coloured-spans aesthetic of the
inspiration screenshot. Body text is dark gray by default; only words
relevant to the row's feature(s) are coloured. For the composite row,
Portuguese-only words are blue, formal-marker words are orange, and words
that are *both* Portuguese and clearly formal are purple (mix).
"""
import os, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_PNG = 'writeup/figures/composition_formal_pt.png'
OUT_PDF = 'writeup/figures/composition_formal_pt.pdf'

# Colours
C_TEXT = '#2a2a2a'        # default body
C_LABEL_NEUTRAL = '#222222'      # prompt and unsteered share this
C_FORMAL = '#D9822B'      # orange — formal English
C_PT     = '#1F77B4'      # blue — Portuguese
C_BOTH   = '#7B3294'      # purple — Portuguese AND formal

PROMPT = "she's literally the worst, omg"

# Each row: (label, label_colour, segments)
# segments is a list of (text, colour) tokens; "\n" in text creates paragraph breaks.
ROWS = [
    (
        'Unsteered',
        C_LABEL_NEUTRAL,
        [
            ("i hate her. Oh, you're still alive. I can't stand her. "
             "She is the worst. Ugh i hate her. What is the point of her?",
             C_TEXT),
        ],
    ),
    (
        '+ Formal  (30°)',
        C_FORMAL,
        [
            ("i hate her. i ", C_TEXT),
            ("despise", C_FORMAL),
            (" her. i ", C_TEXT),
            ("abhor", C_FORMAL),
            (" her. ", C_TEXT),
            ("the lady", C_FORMAL),
            (" is so ", C_TEXT),
            ("revolting", C_FORMAL),
            (". ", C_TEXT),
            ("the lady", C_FORMAL),
            (" is so ", C_TEXT),
            ("repugnant", C_FORMAL),
            (".", C_TEXT),
        ],
    ),
    (
        '+ Portuguese  (20°)',
        C_PT,
        [
            ("the way she's always trying to make herself the victim and "
             "blame others for how she feels… and then she gets mad ",
             C_TEXT),
            ("quando você não quer falar com ela e quer ficar "
             "sozinha. não sei pq ela pensa que eu vou querer falar "
             "com ela, se ela nunca me pergunta como eu estou.", C_PT),
        ],
    ),
    (
        '+ Formal + Portuguese  (20° each)',
        C_BOTH,
        [
            ("i can ", C_TEXT),
            ("até entender a opinião dela se o filme não tivesse sido "
             "premiado, não tivesse arrecaddo o valor que arrecadou, "
             "e se os críticos não tivessem elogiado o filme. "
             "contudo, considerando o sucesso obtido e os elogios "
             "recebidos, é bastante desproporcional e desproporcionada "
             "a sua crítica.", C_BOTH),
        ],
    ),
]

# ---- Layout ----
FIG_W_IN = 11.5
FONT_PROMPT_LABEL = 11
FONT_LABEL = 10.5
FONT_BODY = 10
FONT_FAMILY = 'monospace'
WRAP_CHARS = 88            # body wrap width (chars at 10pt mono)
LABEL_COL_W = 3.5          # inches (label col)
LEFT_MARGIN = 0.45
RIGHT_MARGIN = 0.45
TOP_MARGIN = 0.55
BOTTOM_MARGIN = 0.45
PROMPT_GAP = 0.30
LABEL_BODY_GAP = 0.10      # vertical between label-line and body-line
ROW_GAP = 0.30             # between rows
LINE_HEIGHT = FONT_BODY * 1.4 / 72      # inches
PARA_GAP = 0.05            # between paragraphs within a row


def fragment_to_tokens(text, color):
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
    lines = []
    current = []
    cur_len = 0
    i = 0
    while i < len(tokens):
        tok, col = tokens[i]
        if tok == '\n':
            n = 1
            while i + 1 < len(tokens) and tokens[i + 1][0] == '\n':
                n += 1; i += 1
            if current:
                lines.append(current); current = []; cur_len = 0
            if n >= 2:
                lines.append('PARA')
            i += 1; continue
        if tok == ' ':
            if current:
                current.append((' ', col)); cur_len += 1
            i += 1; continue
        if cur_len + len(tok) > max_chars and current:
            while current and current[-1][0] == ' ':
                current.pop()
            lines.append(current); current = []; cur_len = 0
        current.append((tok, col))
        cur_len += len(tok)
        i += 1
    if current:
        lines.append(current)
    return lines


def main():
    # First pass: wrap all rows + prompt to compute height
    prompt_lines = wrap_tokens(
        fragment_to_tokens(PROMPT, C_TEXT), WRAP_CHARS)
    rows_lines = []
    for _, _, segs in ROWS:
        toks = []
        for t, c in segs:
            toks.extend(fragment_to_tokens(t, c))
        rows_lines.append(wrap_tokens(toks, WRAP_CHARS))

    # Compute total height in inches
    total_h = TOP_MARGIN + BOTTOM_MARGIN
    # Prompt
    total_h += LINE_HEIGHT  # label-line of prompt
    for ln in prompt_lines:
        if ln == 'PARA':
            total_h += PARA_GAP
        else:
            total_h += LINE_HEIGHT
    total_h += PROMPT_GAP
    # Each row: body lines (label inline with first) + ROW_GAP
    for lines in rows_lines:
        for ln in lines:
            if ln == 'PARA':
                total_h += PARA_GAP
            else:
                total_h += LINE_HEIGHT
        total_h += ROW_GAP
    # Trim trailing ROW_GAP
    total_h -= ROW_GAP

    fig = plt.figure(figsize=(FIG_W_IN, total_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W_IN)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.axis('off')

    # Outer rounded box
    box = FancyBboxPatch(
        (0.18, 0.18), FIG_W_IN - 0.36, total_h - 0.36,
        boxstyle='round,pad=0.0,rounding_size=0.18',
        linewidth=1.2, edgecolor='#888888', facecolor='white')
    ax.add_patch(box)

    label_x = LEFT_MARGIN
    body_x = LEFT_MARGIN + LABEL_COL_W

    y = TOP_MARGIN

    # --- Prompt block ---
    ax.text(label_x, y, 'Prompt:', fontsize=FONT_PROMPT_LABEL,
            fontfamily=FONT_FAMILY, fontweight='bold',
            color=C_LABEL_NEUTRAL, va='top')
    # Render prompt body inline starting at body_x
    x = body_x
    for tok, col in fragment_to_tokens(PROMPT, C_TEXT):
        if tok == ' ':
            t = ax.text(x, y, ' ', fontsize=FONT_BODY,
                        fontfamily=FONT_FAMILY, color=col, va='top')
            fig.canvas.draw()
            bb = t.get_window_extent()
            x += bb.width / fig.dpi
        elif tok == '\n':
            y += LINE_HEIGHT; x = body_x
        else:
            t = ax.text(x, y, tok, fontsize=FONT_BODY,
                        fontfamily=FONT_FAMILY, color=col, va='top')
            fig.canvas.draw()
            bb = t.get_window_extent()
            x += bb.width / fig.dpi
    y += LINE_HEIGHT
    y += PROMPT_GAP

    # --- Each row (label inline with first body line, hanging indent) ---
    for (label, label_color, _), lines in zip(ROWS, rows_lines):
        # Draw label at label_x, same y as first body line.
        ax.text(label_x, y, label + ':', fontsize=FONT_LABEL,
                fontfamily=FONT_FAMILY, fontweight='bold',
                color=label_color, va='top')
        for ln in lines:
            if ln == 'PARA':
                y += PARA_GAP; continue
            x = body_x
            for tok, col in ln:
                t = ax.text(x, y, tok if tok != ' ' else ' ',
                            fontsize=FONT_BODY,
                            fontfamily=FONT_FAMILY, color=col, va='top')
                fig.canvas.draw()
                bb = t.get_window_extent()
                x += bb.width / fig.dpi
            y += LINE_HEIGHT
        y += ROW_GAP

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {OUT_PNG}')
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
