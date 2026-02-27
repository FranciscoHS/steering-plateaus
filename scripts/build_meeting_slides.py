#!/usr/bin/env python3
"""Build new meeting slides and insert them after slide 1 of the existing deck."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pathlib import Path
import copy
from lxml import etree

# --- Paths ---
DECK = Path("/home/francisco/Research/steering-plateaus/Activation Plateaus Meeting Slides.pptx")
VAULT = Path("/home/francisco/Documents/Francisco's Vault")
RESULTS = Path("/home/francisco/Research/steering-plateaus/results")
WRITEUP = Path("/home/francisco/Research/steering-plateaus/writeup")
OUTPUT = Path("/home/francisco/Research/steering-plateaus/Activation Plateaus Meeting Slides - Updated.pptx")

# Colors matching existing deck (Google Slides defaults)
TITLE_COLOR = RGBColor(0, 0, 0)
BODY_COLOR = RGBColor(0x59, 0x59, 0x59)  # dark gray
BOLD_COLOR = RGBColor(0x33, 0x33, 0x33)


def img(name):
    """Find an image, checking vault first then results."""
    p = VAULT / name
    if p.exists():
        return str(p)
    # Search results recursively
    for f in RESULTS.rglob(name):
        return str(f)
    for f in WRITEUP.rglob(name):
        return str(f)
    print(f"  WARNING: Image not found: {name}")
    return None


def add_title(slide, text, font_size=Pt(28)):
    """Add a title to a slide using placeholder 0 if available, else a text box."""
    if slide.placeholders and 0 in {p.placeholder_format.idx for p in slide.placeholders}:
        title = slide.placeholders[0]
        title.text = text
        for para in title.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = font_size
                run.font.color.rgb = TITLE_COLOR
    else:
        txBox = slide.shapes.add_textbox(Inches(0.3), Inches(0.4), Inches(9.3), Inches(0.7))
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = text
        run.font.size = font_size
        run.font.color.rgb = TITLE_COLOR


def make_textbox(slide, left, top, width, height, bullets, font_size=Pt(14)):
    """Add a text box with bullet points."""
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        # Handle bold prefix (text before first colon or **markers**)
        if isinstance(item, tuple):
            # (bold_part, normal_part)
            bold_run = p.add_run()
            bold_run.text = item[0]
            bold_run.font.size = font_size
            bold_run.font.bold = True
            bold_run.font.color.rgb = BODY_COLOR
            normal_run = p.add_run()
            normal_run.text = item[1]
            normal_run.font.size = font_size
            normal_run.font.color.rgb = BODY_COLOR
        else:
            run = p.add_run()
            run.text = item
            run.font.size = font_size
            run.font.color.rgb = BODY_COLOR

        p.space_after = Pt(6)


def make_body_text(slide, text_lines, font_size=Pt(16), top=1.2):
    """Add body text (non-bulleted paragraphs)."""
    txBox = slide.shapes.add_textbox(
        Inches(0.4), Inches(top), Inches(9.0), Inches(4.2)
    )
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, line in enumerate(text_lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        if isinstance(line, tuple):
            bold_run = p.add_run()
            bold_run.text = line[0]
            bold_run.font.size = font_size
            bold_run.font.bold = True
            bold_run.font.color.rgb = BODY_COLOR
            if line[1]:
                normal_run = p.add_run()
                normal_run.text = line[1]
                normal_run.font.size = font_size
                normal_run.font.color.rgb = BODY_COLOR
        else:
            run = p.add_run()
            run.text = line
            run.font.size = font_size
            run.font.color.rgb = BODY_COLOR

        p.space_after = Pt(8)


def add_image_safe(slide, img_path, left, top, width=None, height=None):
    """Add an image to a slide if the path exists."""
    if img_path and Path(img_path).exists():
        kwargs = {"image_file": img_path, "left": Inches(left), "top": Inches(top)}
        if width:
            kwargs["width"] = Inches(width)
        if height:
            kwargs["height"] = Inches(height)
        slide.shapes.add_picture(**kwargs)
        return True
    return False


def reorder_slides(prs, new_order):
    """Reorder slides by providing a list of current indices in desired order.

    E.g. new_order=[0, 5, 6, 7, 1, 2, 3, 4] moves slides 5-7 after slide 0.
    """
    sldIdLst = prs.slides._sldIdLst
    els = list(sldIdLst)
    # Remove all, re-add in desired order
    for el in els:
        sldIdLst.remove(el)
    for idx in new_order:
        sldIdLst.append(els[idx])


def build_slides():
    prs = Presentation(str(DECK))
    n_existing = len(prs.slides)
    # We'll use layout indices from the template
    layout_title_body = prs.slide_layouts[2]   # TITLE_AND_BODY
    layout_title_only = prs.slide_layouts[4]   # TITLE_ONLY
    layout_two_col = prs.slide_layouts[3]      # TITLE_AND_TWO_COLUMNS
    layout_blank = prs.slide_layouts[10]       # BLANK

    new_slides_start = n_existing  # new slides will be appended, then reordered

    # ========== SLIDE 2: Contents ==========
    slide = prs.slides.add_slide(layout_title_body)
    add_title(slide, "Meeting 27/02/2026 - Contents")
    make_body_text(slide, [
        ("Last time: ", "multi-contrastive steering, adapted MELBO, auto-interp, discussed research directions."),
        "",
        ("Today:", ""),
        "1. Plateau-ness degrades away from real activations",
        "2. Cross-group feature composition",
        "3. Two dead ends (Jacobian directions, sparse models)",
        "4. Quantitative model for plateau-breaking angles",
        "5. ChessGPT: three-regime validation",
        "6. Subspace scaling \u2014 25 features",
        "7. Feature refinement + progressive discovery (~140 features)",
        "8. Subspace scaling \u2014 135 features with variable thresholds",
        "9. Feature reconstruction",
        "10. Discussion / next steps",
    ], font_size=Pt(13))

    # ========== SLIDE 3: Plateau-ness degrades off-manifold ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Plateau-ness degrades away from real activations")
    add_image_safe(slide, img("plateau_width_vs_steering.png"), 0.3, 1.2, width=4.2)
    add_image_safe(slide, img("kl_curves_by_steering.png"), 4.7, 1.2, width=4.8)
    make_textbox(slide, 0.3, 4.3, 9.3, 0.9, [
        "Steer at increasing magnitudes, measure plateau width. Plateau shrinks. Addresses direction 1.5.",
    ], font_size=Pt(13))

    # ========== SLIDE 4: Cross-group feature composition ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Cross-group feature composition")
    add_image_safe(slide, img("crossgroup_composition_figure.png"), 0.3, 1.2, width=4.2)
    add_image_safe(slide, img("feature_vs_random_sweep_clean.png"), 4.7, 1.2, width=4.8)
    make_textbox(slide, 0.3, 4.3, 9.3, 0.9, [
        "Russian + C = Russian C. Scaling monotonic but weak. Methodological note: must include dominant language.",
    ], font_size=Pt(13))

    # ========== SLIDE 5: Two dead ends ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Two dead ends")
    # Left column
    make_textbox(slide, 0.3, 1.3, 4.3, 3.0, [
        ("Jacobian-norm directions aren't features", ""),
    ], font_size=Pt(14))
    # Right column
    make_textbox(slide, 5.2, 1.3, 4.3, 3.0, [
        ("Sparse models lack plateaus", ""),
        "",
        "Pivoted to ChessGPT.",
    ], font_size=Pt(14))
    # Optional figure
    add_image_safe(slide, img("perturbation_v1.png"), 0.3, 2.5, width=4.3)

    # ========== SLIDE 6: Quantitative model setup ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Quantitative model for plateau-breaking")
    make_body_text(slide, [
        ("Key equation: ", "\u03b8_break(d) = arcsin( sin(\u03b8\u2081) / \u03b1(d) )"),
        "  where \u03b1(d) = max_i |<d, f_i>| is the max alignment with any feature.",
        "",
        ("Three predictions:", ""),
        ("  1. Single feature: ", "\u03b8_break = \u03b8\u2081 \u2248 5.5\u00b0"),
        ("  2. Random in k-dim subspace: ", "\u03b8_sub = arcsin( sin(\u03b8\u2081) \u00b7 \u221a(k / 2 ln k) ) \u2248 11\u00b0 for k=25"),
        ("  3. Fully random: ", "\u03b8_rand = arcsin( sin(\u03b8\u2081) \u00b7 \u221a(n / 2 ln m) ) \u2248 36\u00b0 for m\u22481000"),
        "",
        "Brief \u2014 details in the shared write-up.",
    ], font_size=Pt(15))

    # ========== SLIDE 7: Three-regime validation ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "ChessGPT: three-regime validation")
    add_image_safe(slide, img("subspace_vs_random.png"), 0.3, 1.2, width=5.3)
    make_textbox(slide, 5.8, 1.3, 3.8, 3.8, [
        "Observed vs predicted:",
        "",
        ("Features: ", "~5.5\u00b0"),
        ("Subspace random: ", "~12.5\u00b0 (pred: 11\u00b0)"),
        ("Fully random: ", "~36.3\u00b0 (pred: 36\u00b0, m\u22481000)"),
    ], font_size=Pt(14))

    # ========== SLIDE 8: Subspace scaling — 25 features ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Subspace scaling \u2014 25 features")
    add_image_safe(slide, img("subspace_scaling.png"), 0.3, 1.2, width=5.3)
    make_textbox(slide, 5.8, 1.3, 3.8, 3.8, [
        "Directionally correct, underpredicts by 20\u201330%.",
        "",
        ("Two confounders:", ""),
        "1. Dirty features (rotated from true features)",
        "2. Variable \u03b8\u2081 per feature (model assumes uniform)",
    ], font_size=Pt(13))

    # ========== SLIDE 9: Feature refinement + progressive discovery ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Feature refinement + progressive discovery")
    add_image_safe(slide, img("refine_all_features.png"), 0.2, 1.2, width=4.5)
    add_image_safe(slide, img("progressive_discovery.png"), 4.9, 1.2, width=4.8)
    make_textbox(slide, 0.2, 4.2, 4.5, 1.0, [
        "Local optimization cleans features. \u03b8\u2081 range: 2.2\u20135.5\u00b0.",
    ], font_size=Pt(12))
    make_textbox(slide, 4.9, 4.2, 4.8, 1.0, [
        "Progressive MELBO: ~140 features in 512-d space.",
    ], font_size=Pt(12))

    # ========== SLIDE 10: Subspace scaling — 135 features ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Subspace scaling \u2014 135 features, variable thresholds")
    add_image_safe(slide, img("subspace_pred_scaling_cd0.1_simple.png"), 0.3, 1.2, width=5.3)
    make_textbox(slide, 5.8, 1.3, 3.8, 3.8, [
        "Predicted trend increases, empirical is flat, error bars overlap.",
        "",
        "Current status.",
    ], font_size=Pt(14))

    # ========== SLIDE 11: Feature reconstruction ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Feature reconstruction")
    reconstruction_img = img("reconstruction.png")
    if reconstruction_img:
        add_image_safe(slide, reconstruction_img, 0.3, 1.2, width=5.0)
        text_left = 5.5
    else:
        text_left = 0.4
    make_textbox(slide, text_left, 1.3, 9.3 - text_left, 3.8, [
        "62 directions explain 93.4% variance. Top 3 explain 83%.",
        "",
        "Residuals have random-like plateau angles.",
        "",
        "d1 correlates r > 0.96 with move count.",
    ], font_size=Pt(14))

    # ========== SLIDE 12: Discussion / next steps ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Discussion / next steps")
    make_body_text(slide, [
        "Reference research directions document.",
        "",
        "What's prioritized, what's open for discussion.",
    ], font_size=Pt(16))

    # ===== Reorder: [title, new_0, new_1, ..., new_10, old_1, old_2, ...] =====
    n_new = len(prs.slides) - n_existing  # should be 11
    print(f"Added {n_new} new slides. Reordering...")
    # Current indices: 0..n_existing-1 are old, n_existing..end are new
    # Desired order: [0] + [n_existing..end] + [1..n_existing-1]
    new_order = [0] + list(range(n_existing, n_existing + n_new)) + list(range(1, n_existing))
    reorder_slides(prs, new_order)

    prs.save(str(OUTPUT))
    print(f"Saved to {OUTPUT}")
    print(f"Total slides: {len(prs.slides)} ({n_new} new + {n_existing} existing)")


if __name__ == "__main__":
    build_slides()
