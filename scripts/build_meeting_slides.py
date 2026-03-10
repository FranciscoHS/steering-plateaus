#!/usr/bin/env python3
"""Build new meeting slides and insert them after slide 1 of the existing deck."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pathlib import Path
import copy
from lxml import etree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


EQ_DIR = Path("/home/francisco/Research/steering-plateaus/results/equations")


def render_equation(latex_str, name, fontsize=18, dpi=200):
    """Render a LaTeX equation to a transparent PNG image, return its path."""
    EQ_DIR.mkdir(parents=True, exist_ok=True)
    outpath = EQ_DIR / f"{name}.png"
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.set_axis_off()
    ax.text(0, 0.5, f"${latex_str}$", fontsize=fontsize,
            transform=ax.transAxes, va='center')
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight', pad_inches=0.05,
                transparent=True)
    plt.close(fig)
    return str(outpath)


def add_equation(slide, latex_str, left, top, height, name):
    """Render a LaTeX equation as an image and add it to the slide."""
    img_path = render_equation(latex_str, name)
    slide.shapes.add_picture(img_path, Inches(left), Inches(top), height=Inches(height))


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
        ("Today:", ""),
        "1. Plateau-ness degrades away from real activations",
        "2. Cross-group feature composition",
        "3. Jacobian optimizations",
        "4. Quantitative model for plateau-breaking angles",
        "5. ChessGPT: three-regime validation",
        "6. Subspace scaling \u2014 25 features",
        "7. Feature refinement + progressive discovery (~140 features)",
        "8. Subspace scaling \u2014 135 features with variable thresholds",
        "9. Feature reconstruction",
        "10. Discussion / next steps",
    ], font_size=Pt(14))

    # ========== SLIDE 3: Plateau-ness degrades away from real activations ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Plateau-ness degrades away from real activations")
    add_image_safe(slide, img("plateau_width_vs_steering.png"), 0.3, 1.2, width=4.2)
    add_image_safe(slide, img("kl_curves_by_steering.png"), 4.8, 1.2, width=4.7)
    make_textbox(slide, 0.3, 4.5, 9.3, 0.7, [
        "\u2022 Steer at increasing magnitudes, measure plateau width \u2192 plateau shrinks",
    ], font_size=Pt(13))

    # ========== SLIDE 4: Cross-group feature composition ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Cross-group feature composition")
    add_image_safe(slide, img("crossgroup_composition_figure.png"), 0.3, 1.2, width=4.2)
    add_image_safe(slide, img("feature_vs_random_sweep_clean.png"), 4.8, 1.2, width=4.7)
    make_textbox(slide, 0.3, 4.5, 9.3, 0.7, [
        "\u2022 Russian + C = Russian C. Scaling monotonic but weak",
        "\u2022 Methodological note: must include dominant language in multi-contrastive set",
    ], font_size=Pt(12))

    # ========== SLIDE 5: Jacobian optimizations ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Jacobian Optimizations")
    make_textbox(slide, 0.3, 1.2, 4.8, 3.2, [
        ("\u2022 Approach: ", "optimize for directions that minimize Jacobian norm of resulting activations"),
        "\u2022 Result: found directions producing real-ish activations with plateaus",
        ("\u2022 But: ", "perturbation plots along these directions look random (right)"),
        "\u2022 Steering: minimal changes, then model breaks down",
        "",
        ("\u2022 Conclusion: ", "optimizers can find directions producing similar activations that aren't features"),
    ], font_size=Pt(12))
    add_image_safe(slide, img("perturbation_v1.png"), 5.3, 1.2, width=4.2)

    # ========== SLIDE 6: Quantitative model setup ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Quantitative model for plateau-breaking")

    # Key equation as rendered image
    make_textbox(slide, 0.4, 1.2, 3.0, 0.4, ["\u2022 Key equation:"], font_size=Pt(14))
    add_equation(slide,
        r"\theta_{\mathrm{break}}(d) = \arcsin\!\left(\frac{\sin(\theta_1)}{\alpha(d)}\right)",
        3.5, 1.1, 0.45, "eq_main")

    make_textbox(slide, 0.6, 1.7, 8.8, 0.4, [
        "where \u03b1(d) = max alignment of d with any feature",
    ], font_size=Pt(12))

    # Three predictions with equations
    make_textbox(slide, 0.4, 2.2, 9.0, 0.4, [
        ("\u2022 Three predictions:", ""),
    ], font_size=Pt(14))

    # Prediction 1
    make_textbox(slide, 0.6, 2.6, 3.5, 0.4, ["1. Single feature:"], font_size=Pt(13))
    add_equation(slide, r"\theta_{\mathrm{break}} = \theta_1 \approx 5.5°",
        4.0, 2.55, 0.35, "eq_pred1")

    # Prediction 2
    make_textbox(slide, 0.6, 3.1, 4.0, 0.4, ["2. Random in k-dim subspace:"], font_size=Pt(13))
    add_equation(slide,
        r"\theta_{\mathrm{sub}} \approx \arcsin\!\left(\sin(\theta_1) \cdot \sqrt{\frac{k}{2\ln k}}\right) \approx 11°",
        0.8, 3.45, 0.45, "eq_pred2")

    # Prediction 3
    make_textbox(slide, 0.6, 4.0, 3.5, 0.4, ["3. Fully random:"], font_size=Pt(13))
    add_equation(slide,
        r"\theta_{\mathrm{rand}} \approx \arcsin\!\left(\sin(\theta_1) \cdot \sqrt{\frac{n}{2\ln m}}\right) \approx 36°",
        0.8, 4.35, 0.45, "eq_pred3")

    make_textbox(slide, 0.4, 4.9, 9.0, 0.4, [
        "\u2022 Details in the shared write-up",
    ], font_size=Pt(14))

    # ========== SLIDE 7: Three-regime validation ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "ChessGPT: three-regime validation")
    add_image_safe(slide, img("subspace_vs_random.png"), 0.3, 1.2, width=5.0)
    make_textbox(slide, 5.5, 1.3, 4.1, 3.8, [
        ("Observed vs predicted:", ""),
        "",
        ("\u2022 Features: ", "~5.5\u00b0"),
        ("\u2022 Subspace random: ", "~12.5\u00b0 (pred: 11\u00b0)"),
        ("\u2022 Fully random: ", "~36.3\u00b0 (pred: 36\u00b0, m\u22481000)"),
    ], font_size=Pt(14))

    # ========== SLIDE 8: Subspace scaling — 25 features ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Subspace scaling \u2014 25 features")
    add_image_safe(slide, img("subspace_scaling.png"), 0.3, 1.2, width=5.0)
    make_textbox(slide, 5.5, 1.3, 4.1, 3.8, [
        "\u2022 Directionally correct, underpredicts by 20\u201330%",
        "",
        ("\u2022 Two confounders:", ""),
        "    1. Dirty features (rotated from true features)",
        "    2. Variable \u03b8\u2081 per feature (model assumes uniform)",
    ], font_size=Pt(13))

    # ========== SLIDE 9: Feature refinement + progressive discovery ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Feature refinement + progressive discovery")
    add_image_safe(slide, img("refine_all_features.png"), 0.2, 1.2, width=4.4)
    add_image_safe(slide, img("progressive_discovery.png"), 4.8, 1.2, width=4.7)
    make_textbox(slide, 0.2, 4.4, 4.4, 0.8, [
        "\u2022 Local optimization cleans features. \u03b8\u2081 range: 2.2\u20135.5\u00b0",
    ], font_size=Pt(12))
    make_textbox(slide, 4.8, 4.4, 4.7, 0.8, [
        "\u2022 Progressive MELBO: ~140 features in 512-d space",
    ], font_size=Pt(12))

    # ========== SLIDE 10: Subspace scaling — 135 features ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Subspace scaling \u2014 135 features, variable thresholds")
    add_image_safe(slide, img("subspace_pred_scaling_cd0.1_simple.png"), 0.3, 1.2, width=5.0)
    make_textbox(slide, 5.5, 1.3, 4.1, 3.8, [
        "\u2022 Predicted trend increases, empirical is flat",
        "\u2022 Error bars overlap but trends disagree",
        "",
        "\u2022 Current status",
    ], font_size=Pt(14))

    # ========== SLIDE 11: Feature reconstruction ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Feature reconstruction")
    reconstruction_img = img("reconstruction.png")
    if reconstruction_img:
        add_image_safe(slide, reconstruction_img, 0.3, 1.2, width=4.8)
        text_left = 5.3
    else:
        text_left = 0.4
    make_textbox(slide, text_left, 1.3, 9.3 - text_left, 3.8, [
        "\u2022 62 directions explain 93.4% variance. Top 3 explain 83%",
        "",
        "\u2022 Most residual plateau-breaking angles close to random baseline, but some outliers",
        "",
        "\u2022 Outliers suggest we can extract more features from residuals",
        "",
        "\u2022 d\u2081 correlates r > 0.96 with move count",
    ], font_size=Pt(13))

    # ========== SLIDE 12: Discussion / next steps ==========
    slide = prs.slides.add_slide(layout_title_only)
    add_title(slide, "Discussion / next steps")
    make_body_text(slide, [
        "\u2022 Reference research directions document",
        "",
        "\u2022 What's prioritized, what's open for discussion",
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
