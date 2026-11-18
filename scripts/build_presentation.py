"""
build_presentation.py
Generates a 16:9 presentation on Explainability-Aware Synthetic Oversampling (SHAP-SMOTE).
Includes slide contents, cards, visual layouts, embedded figures, and speaker notes.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Initialize Presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# Theme Colors
C_DARK_BG = RGBColor(15, 23, 42)      # Slate 900
C_LIGHT_BG = RGBColor(248, 250, 252)  # Slate 50
C_CARD_BG = RGBColor(255, 255, 255)   # White
C_BORDER = RGBColor(226, 232, 240)    # Slate 200
C_TEXT_DARK = RGBColor(15, 23, 42)    # Slate 900
C_TEXT_MUTED = RGBColor(71, 85, 105)  # Slate 600
C_TEXT_LIGHT = RGBColor(241, 245, 249)# Slate 100
C_PRIMARY = RGBColor(37, 99, 235)     # Blue 600
C_PRIMARY_DARK = RGBColor(29, 78, 216)# Blue 700
C_SECONDARY = RGBColor(13, 148, 136)  # Teal 600
C_ACCENT_WARN = RGBColor(220, 38, 38) # Red 600
C_ACCENT_GOLD = RGBColor(217, 119, 6) # Amber 600
C_BADGE_BG = RGBColor(238, 242, 255)  # Indigo 50
C_BADGE_TEXT = RGBColor(67, 56, 202)  # Indigo 700

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def set_slide_background(slide, color):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_header(slide, title_text, category="RESEARCH & BENCHMARK"):
    # Category badge
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(10), Inches(0.35))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = C_PRIMARY

    # Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.7))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

def add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape

def add_speaker_note(slide, note_text):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = note_text

# ==========================================================
# SLIDE 1: Title Slide (Dark Theme)
# ==========================================================
s1 = prs.slides.add_slide(blank_layout)
set_slide_background(s1, C_DARK_BG)

# Title container
tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(3.0))
tf = tbox.text_frame
tf.word_wrap = True

p_sub_tag = tf.paragraphs[0]
p_sub_tag.text = "RESEARCH & EMPIRICAL BENCHMARK"
p_sub_tag.font.size = Pt(13)
p_sub_tag.font.bold = True
p_sub_tag.font.color.rgb = RGBColor(96, 165, 250)
p_sub_tag.space_after = Pt(14)

p_main = tf.add_paragraph()
p_main.text = "Explainability-Aware Synthetic Oversampling\nfor Imbalanced Tabular Classification"
p_main.font.size = Pt(36)
p_main.font.bold = True
p_main.font.color.rgb = RGBColor(255, 255, 255)
p_main.space_after = Pt(16)

p_sub = tf.add_paragraph()
p_sub.text = "A Zero-Data-Leakage Benchmark of SHAP-Guided SMOTE Variants"
p_sub.font.size = Pt(18)
p_sub.font.color.rgb = RGBColor(203, 213, 225)

# Authors / Presenters Box
card_s1 = add_card(s1, Inches(1.0), Inches(4.8), Inches(11.3), Inches(1.8), bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
abox = s1.shapes.add_textbox(Inches(1.2), Inches(4.9), Inches(10.9), Inches(1.6))
tf_a = abox.text_frame
tf_a.word_wrap = True

p_auth_title = tf_a.paragraphs[0]
p_auth_title.text = "AUTHORS & RESEARCH TEAM"
p_auth_title.font.size = Pt(11)
p_auth_title.font.bold = True
p_auth_title.font.color.rgb = RGBColor(148, 163, 184)
p_auth_title.space_after = Pt(6)

p_auth = tf_a.add_paragraph()
p_auth.text = "Rachit Ingole  •  Harshal Jaiswal  •  Hasan Rupawalla  •  Pranav Jagtap\nUnder the Guidance of: Prof. Gopal Deshmukh"
p_auth.font.size = Pt(14)
p_auth.font.bold = True
p_auth.font.color.rgb = RGBColor(241, 245, 249)
p_auth.space_after = Pt(4)

p_inst = tf_a.add_paragraph()
p_inst.text = "Department of Computer Engineering, Vishwakarma Institute of Technology (VIT), Pune, India"
p_inst.font.size = Pt(12)
p_inst.font.color.rgb = RGBColor(148, 163, 184)

add_speaker_note(s1, 
    "Good morning/afternoon everyone. Today, we present our research on 'Explainability-Aware Synthetic Oversampling "
    "for Imbalanced Tabular Classification'. In this work, we investigate whether feature attributions from Explainable AI (TreeSHAP) "
    "can be introduced directly upstream during dataset balancing to guide synthetic sample generation, tested under a strict zero-data-leakage protocol."
)

# ==========================================================
# SLIDE 2: Problem Statement & Real-World High-Stakes Dilemma
# ==========================================================
s2 = prs.slides.add_slide(blank_layout)
set_slide_background(s2, C_LIGHT_BG)
add_header(s2, "The High-Stakes Class Imbalance Dilemma in Tabular Data", "1. Problem Formulation")

# 3 Domain Cards
domains = [
    ("Financial Fraud Detection", "Imbalance Ratio: < 0.5% fraud\n• Massive volume of legitimate transactions.\n• False negatives mean undetected financial fraud and huge financial losses.\n• Linear models fail to capture complex evasion patterns.", RGBColor(239, 246, 255), C_PRIMARY),
    ("Clinical Disease Diagnostics", "Imbalance Ratio: 5% to 10% positive cases\n• Stroke, malignancy, rare pathogen detection.\n• Missing a sick patient is catastrophic (fatal clinical hazard).\n• Requires strict probability calibration for risk scoring.", RGBColor(240, 253, 244), C_SECONDARY),
    ("Cybersecurity & Intrusion", "Imbalance Ratio: ~ 10:1 to 50:1\n• Network attack packets buried in normal traffic.\n• Heavy class overlap and adversarial disguise.\n• Boundary confusion leads to critical alert fatigue.", RGBColor(254, 242, 242), C_ACCENT_WARN),
]

for idx, (title, desc, bg, accent) in enumerate(domains):
    left = Inches(0.8 + idx * 3.95)
    add_card(s2, left, Inches(1.6), Inches(3.8), Inches(3.6), bg_color=bg, border_color=accent)
    box = s2.shapes.add_textbox(left + Inches(0.2), Inches(1.8), Inches(3.4), Inches(3.2))
    tf = box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = title
    p1.font.size = Pt(15)
    p1.font.bold = True
    p1.font.color.rgb = accent
    p1.space_after = Pt(12)
    p2 = tf.add_paragraph()
    p2.text = desc
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_TEXT_DARK

# Bottom Takeaway Card
add_card(s2, Inches(0.8), Inches(5.4), Inches(11.7), Inches(1.5), bg_color=C_CARD_BG, border_color=C_BORDER)
tbox = s2.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(11.3), Inches(1.3))
tf = tbox.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "The Fundamental Challenge for Machine Learning Learners"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.space_after = Pt(4)
p_b = tf.add_paragraph()
p_b.text = "Standard empirical risk minimization minimizes average loss, incentivizing models to simply guess the majority class. Data-level oversampling is mandatory, but classical approaches suffer from severe geometric flaws."
p_b.font.size = Pt(12)
p_b.font.color.rgb = C_TEXT_MUTED

add_speaker_note(s2, 
    "Class imbalance is ubiquitous in high-stakes domains like fraud detection, healthcare diagnostics, and network security. "
    "When a classifier is trained on raw imbalanced data, it minimizes standard loss functions by biasing towards the majority class. "
    "In clinical diagnosis, a false negative can be fatal. Data-level balancing via synthetic oversampling is the standard remedy, "
    "but the way synthetic samples are conventionally generated introduces a major geometric vulnerability."
)

# ==========================================================
# SLIDE 3: The Flaw in Conventional SMOTE (Geometric Breakdown)
# ==========================================================
s3 = prs.slides.add_slide(blank_layout)
set_slide_background(s3, C_LIGHT_BG)
add_header(s3, "The Fundamental Flaw in Classical SMOTE: Feature Agnosticism", "2. Theoretical Motivation")

# Left Column: Explanation Cards
add_card(s3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), bg_color=C_CARD_BG, border_color=C_BORDER)
box = s3.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.9))
tf = box.text_frame
tf.word_wrap = True

p1 = tf.paragraphs[0]
p1.text = "Standard SMOTE Formulation (Chawla et al., 2002)"
p1.font.size = Pt(14)
p1.font.bold = True
p1.font.color.rgb = C_PRIMARY
p1.space_after = Pt(8)

p2 = tf.add_paragraph()
p2.text = "x_new = x_i + λ * (x_zi - x_i),   where λ ~ U(0, 1)\nand x_zi is one of the k-nearest minority neighbors."
p2.font.size = Pt(12)
p2.font.bold = True
p2.font.color.rgb = C_TEXT_DARK
p2.space_after = Pt(14)

p3 = tf.add_paragraph()
p3.text = "Why Euclidean Distance Breaks Down in Tabular Data:"
p3.font.size = Pt(13)
p3.font.bold = True
p3.font.color.rgb = C_ACCENT_WARN
p3.space_after = Pt(6)

bullets = [
    "Isotropic Metric: d(a, b) = sqrt(Σ (a_j - b_j)²). All features contribute equally, regardless of signal strength.",
    "Noise Dominance: As non-informative or weakly correlated features increase, Euclidean distance reflects noise rather than true class manifolds.",
    "Boundary Crossing: SMOTE blindly interpolates across the Bayes decision boundary into majority territory, creating severe label noise."
]
for b in bullets:
    p_bullet = tf.add_paragraph()
    p_bullet.text = "• " + b
    p_bullet.font.size = Pt(11)
    p_bullet.font.color.rgb = C_TEXT_MUTED
    p_bullet.space_after = Pt(6)

# Right Column: Embed Proof Figure 1
fig1_path = os.path.join(REPO_ROOT, "docs", "images", "proofs", "fig1_geometric_decision_boundary_proof.png")
if os.path.exists(fig1_path):
    s3.shapes.add_picture(fig1_path, Inches(6.6), Inches(1.6), width=Inches(5.9))

add_speaker_note(s3,
    "Standard SMOTE identifies neighbors using unweighted Euclidean distance. "
    "In tabular data, many features are noisy or uninformative. When Euclidean distance sums squared differences equally across all dimensions, "
    "noise dimensions dominate. As demonstrated in Figure 1, this causes SMOTE to draw interpolation segments straight across the Bayes boundary "
    "into majority territory, generating synthetic points that corrupt model training."
)

# ==========================================================
# SLIDE 4: The Paradigm Shift: Upstream XAI vs. Post-Hoc XAI
# ==========================================================
s4 = prs.slides.add_slide(blank_layout)
set_slide_background(s4, C_LIGHT_BG)
add_header(s4, "The Paradigm Shift: Moving Explainable AI Upstream", "3. Core Innovation")

# Card 1: Conventional Post-Hoc XAI
add_card(s4, Inches(0.8), Inches(1.6), Inches(5.6), Inches(4.0), bg_color=RGBColor(254, 242, 242), border_color=C_ACCENT_WARN)
b1 = s4.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(3.6))
tf1 = b1.text_frame
tf1.word_wrap = True
p = tf1.paragraphs[0]
p.text = "CONVENTIONAL PARADIGM: Post-Hoc Diagnostic"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_ACCENT_WARN
p.space_after = Pt(10)

flow1 = (
    "1. Imbalanced Raw Data\n"
    "    ↓\n"
    "2. Standard Feature-Agnostic SMOTE (Euclidean)\n"
    "    ↓\n"
    "3. Train Downstream Classification Model\n"
    "    ↓\n"
    "4. Post-Hoc SHAP Explanations (Passive Afterthought)"
)
p_flow1 = tf1.add_paragraph()
p_flow1.text = flow1
p_flow1.font.size = Pt(12)
p_flow1.font.color.rgb = C_TEXT_DARK
p_flow1.space_after = Pt(10)

p_lim = tf1.add_paragraph()
p_lim.text = "Limitation: Explanations are used only after training to diagnose predictions. They do not fix corrupted synthetic samples generated upstream."
p_lim.font.size = Pt(11)
p_lim.font.color.rgb = C_TEXT_MUTED

# Card 2: Proposed Upstream Explainability-Aware Balancing
add_card(s4, Inches(6.9), Inches(1.6), Inches(5.6), Inches(4.0), bg_color=RGBColor(238, 242, 255), border_color=C_PRIMARY)
b2 = s4.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.2), Inches(3.6))
tf2 = b2.text_frame
tf2.word_wrap = True
p = tf2.paragraphs[0]
p.text = "OUR PROPOSED PARADIGM: Upstream Attribution Guidance"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.space_after = Pt(10)

flow2 = (
    "1. Training Fold Only (Zero Leakage)\n"
    "    ↓\n"
    "2. Preliminary Tree Estimator → TreeSHAP Extraction\n"
    "    ↓\n"
    "3. Mean Absolute SHAP Values (MASV) Weights\n"
    "    ↓\n"
    "4. SHAP-Guided Resampling (Geometry / Subspace)\n"
    "    ↓\n"
    "5. Train Downstream Final Classifiers (LR, RF, XGB)"
)
p_flow2 = tf2.add_paragraph()
p_flow2.text = flow2
p_flow2.font.size = Pt(12)
p_flow2.font.color.rgb = C_TEXT_DARK
p_flow2.space_after = Pt(10)

p_adv = tf2.add_paragraph()
p_adv.text = "Core Thesis: Game-theoretic feature attributions guide neighbor discovery along true predictive manifolds."
p_adv.font.size = Pt(11)
p_adv.font.color.rgb = C_PRIMARY_DARK

# Bottom Summary Card
add_card(s4, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.2), bg_color=C_CARD_BG, border_color=C_BORDER)
b3 = s4.shapes.add_textbox(Inches(1.0), Inches(5.9), Inches(11.3), Inches(1.0))
tf3 = b3.text_frame
tf3.word_wrap = True
p_bot = tf3.paragraphs[0]
p_bot.text = "The Key Research Question:"
p_bot.font.size = Pt(13)
p_bot.font.bold = True
p_bot.font.color.rgb = C_PRIMARY
p_b_text = tf3.add_paragraph()
p_b_text.text = "Can Shapley values computed inside the training fold guide synthetic oversampling without leaking information or distorting calibration?"
p_b_text.font.size = Pt(12)
p_b_text.font.color.rgb = C_TEXT_DARK

add_speaker_note(s4,
    "Almost all existing machine learning literature treats SHAP as an after-the-fact diagnostic. "
    "In this project, we invert the pipeline: we fit an initial base estimator on the training fold, compute Mean Absolute SHAP Values (MASV), "
    "and use these game-theoretic attributions to weight or filter the feature space before oversampling takes place."
)

# ==========================================================
# SLIDE 5: Taxonomy of Explainability Touchpoints (T1 - T4)
# ==========================================================
s5 = prs.slides.add_slide(blank_layout)
set_slide_background(s5, C_LIGHT_BG)
add_header(s5, "Formal Taxonomy: 4 Touchpoints of Explainability in Data Balancing", "Objective 1: Taxonomy")

touchpoints = [
    ("T1: Post-Hoc Diagnostic", "Conventional Practice", "Evaluate feature attributions on a model trained on balanced data to diagnose behavior and verify cross-fold stability.", RGBColor(241, 245, 249), C_BORDER),
    ("T2: Pre-Balancing Feature Filtering", "SHAP-Filtered SMOTE (Ours)", "Rank features by MASV, prune uninformative noise dimensions satisfying cumulative threshold θ, search k-NN in informative subspace.", RGBColor(238, 242, 255), C_PRIMARY),
    ("T3: In-Process Metric Weighting", "SHAP-Weighted SMOTE (Ours)", "Directly modulate the distance metric tensor using normalized MASV weights: d_W(a,b) = sqrt(Σ w_j (a_j - b_j)²).", RGBColor(240, 253, 244), C_SECONDARY),
    ("T4: Post-Generation Validation", "Future Extension (e.g. K-IPO)", "Generate candidate synthetic blocks, then screen or reject samples that distort feature-importance rank correlation (Kendall's τ).", RGBColor(254, 243, 199), C_ACCENT_GOLD),
]

for idx, (title, sub, desc, bg, border) in enumerate(touchpoints):
    left = Inches(0.8 + (idx % 2) * 5.95)
    top = Inches(1.6 + (idx // 2) * 2.7)
    add_card(s5, left, top, Inches(5.8), Inches(2.4), bg_color=bg, border_color=border)
    b = s5.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), Inches(5.4), Inches(2.0))
    tf = b.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = title
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = C_TEXT_DARK
    p1.space_after = Pt(2)
    p_sub = tf.add_paragraph()
    p_sub.text = sub
    p_sub.font.size = Pt(11)
    p_sub.font.bold = True
    p_sub.font.color.rgb = C_PRIMARY if "Ours" in sub else C_TEXT_MUTED
    p_sub.space_after = Pt(8)
    p_desc = tf.add_paragraph()
    p_desc.text = desc
    p_desc.font.size = Pt(11)
    p_desc.font.color.rgb = C_TEXT_DARK

add_speaker_note(s5,
    "In Objective 1, we establish a formal theoretical taxonomy categorizing four distinct touchpoints where XAI meets data balancing. "
    "T1 is the standard post-hoc diagnostic. T2 is pre-balancing filtering implemented by SHAP-Filtered SMOTE. "
    "T3 is in-process metric weighting implemented by SHAP-Weighted SMOTE. "
    "T4 is post-generation candidate validation, which we identify as an exciting direction for future work."
)

# ==========================================================
# SLIDE 6: Method 1: SHAP-Weighted SMOTE (MASV-SMOTE)
# ==========================================================
s6 = prs.slides.add_slide(blank_layout)
set_slide_background(s6, C_LIGHT_BG)
add_header(s6, "Method 1: SHAP-Weighted SMOTE (In-Process Metric Weighting)", "Objective 2: Algorithm 1")

# Left Card: Formulation & Math
add_card(s6, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=C_CARD_BG, border_color=C_BORDER)
b_left = s6.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.9))
tf_l = b_left.text_frame
tf_l.word_wrap = True

p = tf_l.paragraphs[0]
p.text = "Mathematical Formulation"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.space_after = Pt(8)

p_eq1 = tf_l.add_paragraph()
p_eq1.text = "1. Mean Absolute SHAP Value (MASV):\n    MASV_j = (1/N) * Σ |φ_j(x_i)|\n\n2. Normalized Feature Weight:\n    w_j = MASV_j / (Σ MASV_m),  where Σ w_j = 1.0\n\n3. Weighted Euclidean Distance:\n    d_W(x_a, x_b) = sqrt( Σ w_j * (x_aj - x_bj)² )"
p_eq1.font.size = Pt(12)
p_eq1.font.bold = True
p_eq1.font.color.rgb = C_TEXT_DARK
p_eq1.space_after = Pt(12)

p_prop = tf_l.add_paragraph()
p_prop.text = "Key Algorithmic Insight:"
p_prop.font.size = Pt(13)
p_prop.font.bold = True
p_prop.font.color.rgb = C_SECONDARY
p_prop.space_after = Pt(4)

p_prop_text = tf_l.add_paragraph()
p_prop_text.text = "Features with high predictive attribution pull nearest neighbors together. Uninformative or noisy dimensions contribute virtually nothing to neighbor discovery."
p_prop_text.font.size = Pt(11)
p_prop_text.font.color.rgb = C_TEXT_MUTED

# Right Card: Algorithmic Efficiency
add_card(s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=RGBColor(240, 253, 244), border_color=C_SECONDARY)
b_right = s6.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
tf_r = b_right.text_frame
tf_r.word_wrap = True

p_algo = tf_r.paragraphs[0]
p_algo.text = "Computational Optimization: Coordinate Warping"
p_algo.font.size = Pt(15)
p_algo.font.bold = True
p_algo.font.color.rgb = C_SECONDARY
p_algo.space_after = Pt(10)

algo_steps = (
    "• Direct weighted distance metrics in Python are computationally expensive (O(N²) brute-force).\n\n"
    "• Our Exact Equivalence Trick:\n"
    "  We transform minority points by the square-root of weights:\n"
    "       Z_min = X_min ⊙ sqrt(w)\n\n"
    "• Standard Euclidean distance on Z_min is mathematically identical to d_W on X_min!\n\n"
    "• Enables optimized C/KD-Tree / Ball-Tree indexing in scikit-learn (O(N log N) query speed).\n\n"
    "• Synthetic interpolation is performed along the original unweighted coordinates: x_new = x_i + λ * (x_z - x_i)."
)
p_steps = tf_r.add_paragraph()
p_steps.text = algo_steps
p_steps.font.size = Pt(11)
p_steps.font.color.rgb = C_TEXT_DARK

add_speaker_note(s6,
    "SHAP-Weighted SMOTE uses Mean Absolute SHAP Values (MASV) to create a weighted metric tensor. "
    "A key algorithmic optimization in our implementation is coordinate warping: by pre-multiplying minority vectors by the square root of weights, "
    "standard Euclidean distance on the transformed space is mathematically identical to the weighted distance, preserving fast KD-tree query speed."
)

# ==========================================================
# SLIDE 7: Method 2: SHAP-Filtered SMOTE (Subspace Variant)
# ==========================================================
s7 = prs.slides.add_slide(blank_layout)
set_slide_background(s7, C_LIGHT_BG)
add_header(s7, "Method 2: SHAP-Filtered SMOTE (Pre-Balancing Subspace Selection)", "Objective 2: Algorithm 2")

# Card 1: Subspace Selection Mechanism
add_card(s7, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=C_CARD_BG, border_color=C_BORDER)
b1 = s7.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.9))
tf1 = b1.text_frame
tf1.word_wrap = True
p = tf1.paragraphs[0]
p.text = "Subspace Pruning Mechanism"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.space_after = Pt(8)

p_math = tf1.add_paragraph()
p_math.text = (
    "1. Sort features by MASV in descending order:\n"
    "     w_(1) ≥ w_(2) ≥ ... ≥ w_(p)\n\n"
    "2. Cumulative Attribution Threshold (θ = 0.85):\n"
    "     p' = min { m : Σ_(r=1)^m w_(r) ≥ θ }\n\n"
    "3. Retain only the minimal p'-dimensional subspace containing 85%+ of predictive energy."
)
p_math.font.size = Pt(12)
p_math.font.bold = True
p_math.font.color.rgb = C_TEXT_DARK
p_math.space_after = Pt(14)

p_why = tf1.add_paragraph()
p_why.text = "Theoretical Benefit: Eliminates the curse of dimensionality during k-NN graph construction by pruning irrelevant noisy attributes."
p_why.font.size = Pt(11)
p_why.font.color.rgb = C_TEXT_MUTED

# Card 2: Interpolation Behavior
add_card(s7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=RGBColor(238, 242, 255), border_color=C_PRIMARY)
b2 = s7.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
tf2 = b2.text_frame
tf2.word_wrap = True
p = tf2.paragraphs[0]
p.text = "Subspace Neighbors vs Full-Space Interpolation"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.space_after = Pt(10)

details = (
    "• Neighbor Graph Construction:\n"
    "  k-nearest neighbor discovery is performed exclusively within the p'-dimensional subspace Z = X_min[:, selected].\n\n"
    "• Full-Dimension Interpolation:\n"
    "  Once the true semantic neighbor index z is identified, the synthetic sample is generated across ALL p features in original space:\n"
    "      s_t = x_i + λ * (x_z - x_i),   λ ~ U(0, 1)\n\n"
    "• Downstream Compatibility:\n"
    "  Ensures the generated synthetic data preserves full dataset dimensionality for downstream classifiers without dropping columns."
)
p_d = tf2.add_paragraph()
p_d.text = details
p_d.font.size = Pt(11)
p_d.font.color.rgb = C_TEXT_DARK

add_speaker_note(s7,
    "SHAP-Filtered SMOTE takes a subspace approach: it sorts features by importance and keeps the smallest subset that accounts for 85% "
    "of the cumulative attribution. Neighbors are found in this reduced, clean subspace, but synthetic samples are interpolated across all features "
    "to ensure complete compatibility with downstream classifiers."
)

# ==========================================================
# SLIDE 8: The Data Leakage Trap (Objective 3)
# ==========================================================
s8 = prs.slides.add_slide(blank_layout)
set_slide_background(s8, C_LIGHT_BG)
add_header(s8, "The Data Leakage Trap: Why Published Oversampling Benchmarks Fail", "Objective 3: Validation")

# Left: Text and proof details
add_card(s8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), bg_color=C_CARD_BG, border_color=C_BORDER)
b = s8.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.9))
tf = b.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "The Catastrophic Fallacy of Global Oversampling"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_ACCENT_WARN
p.space_after = Pt(8)

p_trap = tf.add_paragraph()
p_trap.text = (
    "A pervasive flaw in oversampling literature is executing synthetic generation or feature selection on the FULL dataset prior to cross-validation splits.\n\n"
    "What happens when data leaks?\n"
    "• Synthetic samples are interpolated between train and test instances.\n"
    "• Information from the test fold leaks into the training distribution.\n"
    "• Test samples become near-duplicates of synthetic points."
)
p_trap.font.size = Pt(11)
p_trap.font.color.rgb = C_TEXT_DARK
p_trap.space_after = Pt(12)

# Metric comparison highlight card inside
p_comp = tf.add_paragraph()
p_comp.text = "EMPIRICAL LEAKAGE PROOF (Same Data & Model):"
p_comp.font.size = Pt(12)
p_comp.font.bold = True
p_comp.font.color.rgb = C_TEXT_DARK
p_comp.space_after = Pt(4)

p_scores = tf.add_paragraph()
p_scores.text = "• Leaked Pipeline PR-AUC:  0.9541 (ROC-AUC: 0.9569)\n• Strict Leak-Free PR-AUC: 0.4691 (ROC-AUC: 0.8016)\n• Artificial Inflation:   +103% Fake Boost!"
p_scores.font.size = Pt(12)
p_scores.font.bold = True
p_scores.font.color.rgb = C_ACCENT_WARN

# Right: Embed Proof Figure 2
fig2_path = os.path.join(REPO_ROOT, "docs", "images", "proofs", "fig2_data_leakage_trap_proof.png")
if os.path.exists(fig2_path):
    s8.shapes.add_picture(fig2_path, Inches(6.6), Inches(1.6), width=Inches(5.9))

add_speaker_note(s8,
    "Objective 3 addresses data leakage. In literature, many authors inadvertently run SMOTE before splitting folds. "
    "We ran an empirical diagnostic showing that leaking test instances into oversampling inflates PR-AUC from 0.4691 up to 0.9541—a 103% fake gain. "
    "Our entire benchmarking architecture strictly isolates every operation inside the training fold."
)

# ==========================================================
# SLIDE 9: Leakage-Free Experimental Architecture & Datasets
# ==========================================================
s9 = prs.slides.add_slide(blank_layout)
set_slide_background(s9, C_LIGHT_BG)
add_header(s9, "Leakage-Free Cross-Validation Pipeline & Benchmark Suite", "Objective 3: Protocol")

# Top Card: Pipeline Step Flow
add_card(s9, Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.8), bg_color=C_CARD_BG, border_color=C_BORDER)
b_top = s9.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(11.3), Inches(1.6))
tf_t = b_top.text_frame
tf_t.word_wrap = True

p = tf_t.paragraphs[0]
p.text = "Strict Per-Fold Workflow (5-Fold Stratified Cross-Validation)"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.space_after = Pt(4)

pipeline_flow = (
    "Training Fold (80%) ──> [Fit Scaler] ──> [Fit Base RF (50 trees, depth 5)] ──> [TreeSHAP MASV] ──> [SHAP-SMOTE Resample] ──> [Fit Final Classifier]\n"
    "                                │                                                                                                │\n"
    "Held-Out Test (20%) ───> [Transform Scaler] ──────────────────────────────────────────────────────────────> [Unbiased Multi-Metric Evaluation]"
)
p_fl = tf_t.add_paragraph()
p_fl.text = pipeline_flow
p_fl.font.size = Pt(10)
p_fl.font.bold = True
p_fl.font.color.rgb = C_TEXT_DARK

# Bottom Left: Datasets
add_card(s9, Inches(0.8), Inches(3.6), Inches(5.7), Inches(3.3), bg_color=RGBColor(248, 250, 252), border_color=C_BORDER)
b_ds = s9.shapes.add_textbox(Inches(1.0), Inches(3.8), Inches(5.3), Inches(2.9))
tf_ds = b_ds.text_frame
tf_ds.word_wrap = True
p = tf_ds.paragraphs[0]
p.text = "Diverse Benchmark Datasets Evaluated"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.space_after = Pt(6)

ds_items = (
    "• Breast Cancer Wisconsin Diagnostic (Real): 396 samples, 30 features, 9:1 imbalance ratio.\n"
    "• Simulated Credit Card Fraud: 2,000 samples, 28 features, 49:1 severe imbalance ratio.\n"
    "• Simulated NSL-KDD Intrusion: 1,500 samples, 35 features, 11:1 imbalance with heavy class overlap.\n"
    "• Kaggle Stroke Prediction (Clinical): 5,110 samples, 11 features, 19.5:1 imbalance ratio."
)
p_ds = tf_ds.add_paragraph()
p_ds.text = ds_items
p_ds.font.size = Pt(10)
p_ds.font.color.rgb = C_TEXT_DARK

# Bottom Right: Classifiers & Baselines
add_card(s9, Inches(6.8), Inches(3.6), Inches(5.7), Inches(3.3), bg_color=RGBColor(248, 250, 252), border_color=C_BORDER)
b_cl = s9.shapes.add_textbox(Inches(7.0), Inches(3.8), Inches(5.3), Inches(2.9))
tf_cl = b_cl.text_frame
tf_cl.word_wrap = True
p = tf_cl.paragraphs[0]
p.text = "Classifiers & Baseline Oversamplers"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = C_SECONDARY
p.space_after = Pt(6)

cl_items = (
    "Downstream Classifier Families Evaluated:\n"
    "• L2 Logistic Regression (Linear boundary baseline)\n"
    "• Random Forest (100 trees, depth 6, non-linear ensemble)\n"
    "• XGBoost (100 trees, depth 5, gradient boosting)\n\n"
    "7 Baseline Comparison Methods:\n"
    "• Raw (No Balancing), Random Oversampling (ROS), SMOTE, Borderline-SMOTE, ADASYN, SMOTE-ENN, SMOTE-Tomek."
)
p_cl = tf_cl.add_paragraph()
p_cl.text = cl_items
p_cl.font.size = Pt(10)
p_cl.font.color.rgb = C_TEXT_DARK

add_speaker_note(s9,
    "Here we show our complete experimental setup. Every data-dependent step—scaling, base model training, SHAP calculation, and oversampling—"
    "happens strictly on the training fold. We benchmark across 4 datasets spanning medical, fraud, and cybersecurity domains, "
    "evaluated on 3 distinct classifier families and 7 competitive baseline oversamplers."
)

# ==========================================================
# SLIDE 10: Results: Downstream Minority Utility (PR-AUC & F1)
# ==========================================================
s10 = prs.slides.add_slide(blank_layout)
set_slide_background(s10, C_LIGHT_BG)
add_header(s10, "Downstream Utility Benchmark: PR-AUC and Minority F1-Score", "Objective 4: Results")

# Left Column: Insights Card
add_card(s10, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), bg_color=C_CARD_BG, border_color=C_BORDER)
b_res = s10.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.9))
tf_r = b_res.text_frame
tf_r.word_wrap = True

p = tf_r.paragraphs[0]
p.text = "Key Empirical Performance Findings"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.space_after = Pt(8)

p_table = tf_r.add_paragraph()
p_table.text = (
    "Aggregate Over 9 Dataset-Classifier Configurations:\n"
    "• Unbalanced (Raw):  PR-AUC 0.649  | Minority F1: 0.38*\n"
    "• Standard SMOTE:    PR-AUC 0.601  | Minority F1: 0.564\n"
    "• SHAP-Filtered:     PR-AUC 0.600  | Minority F1: 0.558\n"
    "• SHAP-Weighted:     PR-AUC 0.609  | Minority F1: 0.577\n"
)
p_table.font.size = Pt(11)
p_table.font.bold = True
p_table.font.color.rgb = C_TEXT_DARK
p_table.space_after = Pt(10)

findings = [
    "F1 Advantage: SHAP-Weighted SMOTE achieves higher minority F1 than SMOTE in 6 of 9 configurations (+0.013 overall mean advantage).",
    "Where It Shines (NSL-KDD): On overlapping data, SHAP-Weighted achieves higher F1 across all classifiers (+0.049 on RF, +0.029 on XGBoost).",
    "Score Ranking vs Threshold Trade-off: Raw baseline achieves higher PR-AUC but collapses minority recall to near zero without balancing."
]
for f in findings:
    pf = tf_r.add_paragraph()
    pf.text = "• " + f
    pf.font.size = Pt(10.5)
    pf.font.color.rgb = C_TEXT_MUTED
    pf.space_after = Pt(6)

# Right Column: Embed PR-AUC / F1 Comparison Figure
fig_f1 = os.path.join(REPO_ROOT, "experiments", "results", "figures", "f1_score_comparison.png")
if os.path.exists(fig_f1):
    s10.shapes.add_picture(fig_f1, Inches(6.6), Inches(1.6), width=Inches(5.9))

add_speaker_note(s10,
    "Across the 9 benchmark configurations, SHAP-Weighted SMOTE delivers a higher minority F1 score (0.577 vs 0.564 for standard SMOTE), "
    "beating SMOTE in 6 out of 9 settings. It is especially effective on overlapping datasets like NSL-KDD. "
    "However, we also observe that the raw unbalanced baseline maintains higher PR-AUC, revealing an important trade-off between score ranking "
    "and decision-threshold classification."
)

# ==========================================================
# SLIDE 11: Clinical Benchmark: Kaggle Stroke Dataset
# ==========================================================
s11 = prs.slides.add_slide(blank_layout)
set_slide_background(s11, C_LIGHT_BG)
add_header(s11, "Clinical Benchmark: Kaggle Stroke Prediction Case Study", "Objective 4: Clinical")

# Left Column: Embed Proof Figure 4 (Curves)
fig4_path = os.path.join(REPO_ROOT, "docs", "images", "proofs", "fig4_kaggle_stroke_pr_calibration_curves.png")
if os.path.exists(fig4_path):
    s11.shapes.add_picture(fig4_path, Inches(0.8), Inches(1.6), width=Inches(5.8))

# Right Column: Clinical Benchmark Discussion
add_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=C_CARD_BG, border_color=C_BORDER)
b_clin = s11.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
tf_c = b_clin.text_frame
tf_c.word_wrap = True

p = tf_c.paragraphs[0]
p.text = "Kaggle Stroke Dataset (19.5 : 1 Imbalance)"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.space_after = Pt(8)

p_perf = tf_c.add_paragraph()
p_perf.text = (
    "Clinical Benchmark Performance Summary:\n\n"
    "Method             | PR-AUC | Min Recall | Min F1 | ECE ↓\n"
    "-------------------|--------|------------|--------|-------\n"
    "Raw Baseline       | 0.1834 |   0.0820   | 0.1412 | 0.0382\n"
    "Standard SMOTE     | 0.1742 |   0.2854   | 0.2110 | 0.1624\n"
    "Borderline-SMOTE   | 0.1765 |   0.2714   | 0.2084 | 0.1481\n"
    "SHAP-Filtered      | 0.1812 |   0.2921   | 0.2215 | 0.1580\n"
    "SHAP-Weighted (Ours)| 0.1865|   0.3150   | 0.2341 | 0.1512\n"
)
p_perf.font.size = Pt(10)
p_perf.font.bold = True
p_perf.font.color.rgb = C_TEXT_DARK
p_perf.space_after = Pt(8)

c_bullets = [
    "Clinical Objective: In stroke prediction, missing a patient is fatal. SHAP-Weighted achieves the highest minority recall (0.3150) and highest F1 (0.2341).",
    "Calibration Warning: Oversampling distorts predicted probability calibration (ECE rises from 0.038 to ~0.15), proving that probability recalibration is necessary in clinical use."
]
for cb in c_bullets:
    p_cb = tf_c.add_paragraph()
    p_cb.text = "• " + cb
    p_cb.font.size = Pt(10.5)
    p_cb.font.color.rgb = C_TEXT_MUTED
    p_cb.space_after = Pt(6)

add_speaker_note(s11,
    "On the Kaggle Stroke dataset with a severe 19.5:1 imbalance, missing a positive patient is catastrophic. "
    "SHAP-Weighted SMOTE achieved the highest minority recall of 31.5% and highest F1 of 0.2341. "
    "However, we also observed that all oversamplers increase the Expected Calibration Error (ECE), proving that hospitals cannot directly take "
    "raw oversampled probabilities without downstream probability recalibration."
)

# ==========================================================
# SLIDE 12: Multi-Dimensional Evaluation: Calibration & Stability
# ==========================================================
s12 = prs.slides.add_slide(blank_layout)
set_slide_background(s12, C_LIGHT_BG)
add_header(s12, "Multi-Dimensional Evaluation: Radar Profile & Explanation Stability", "Objective 4: Stability")

# Left Column: Embed Radar Chart
fig_radar = os.path.join(REPO_ROOT, "experiments", "results", "figures", "multi_dimensional_radar.png")
if os.path.exists(fig_radar):
    s12.shapes.add_picture(fig_radar, Inches(0.8), Inches(1.6), width=Inches(5.8))

# Right Column: Explanation Stability (Kendall's Tau)
add_card(s12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=C_CARD_BG, border_color=C_BORDER)
b_stab = s12.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
tf_s = b_stab.text_frame
tf_s.word_wrap = True

p = tf_s.paragraphs[0]
p.text = "Attribution Stability Across Folds (Kendall's τ)"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.space_after = Pt(8)

p_stab_t = tf_s.add_paragraph()
p_stab_t.text = (
    "Stability measures whether the model's explanations remain consistent across different CV folds:\n\n"
    "Dataset         | Raw   | SMOTE | SHAP-F | SHAP-W\n"
    "----------------|-------|-------|--------|--------\n"
    "Breast Cancer   | 0.745 | 0.758 | 0.756  | 0.752\n"
    "Credit Fraud    | 0.580 | 0.606 | 0.600  | 0.614\n"
    "NSL-KDD Style   | 0.507 | 0.602 | 0.503  | 0.577\n"
)
p_stab_t.font.size = Pt(10)
p_stab_t.font.bold = True
p_stab_t.font.color.rgb = C_TEXT_DARK
p_stab_t.space_after = Pt(8)

stab_bullets = [
    "No Universal Stability Advantage: SHAP-Weighted is more stable on Credit Fraud (0.614 vs 0.606) but slightly lower on Breast Cancer (0.752 vs 0.758).",
    "Synthetic Data Fidelity: Wasserstein-1 distances and 2-sample KS tests confirm synthetic samples closely mimic true minority feature distributions."
]
for sb in stab_bullets:
    psb = tf_s.add_paragraph()
    psb.text = "• " + sb
    psb.font.size = Pt(10.5)
    psb.font.color.rgb = C_TEXT_MUTED
    psb.space_after = Pt(6)

add_speaker_note(s12,
    "Beyond classification metrics, Objective 4 examines explanation stability. If a hospital uses a model, its explanations shouldn't wildly shift "
    "based on the data split. Our results show that SHAP-guided oversamplers are competitive with standard SMOTE on Kendall's tau stability, "
    "improving stability on credit fraud but showing comparable stability on medical data."
)

# ==========================================================
# SLIDE 13: Scientific Honesty & Statistical Significance
# ==========================================================
s13 = prs.slides.add_slide(blank_layout)
set_slide_background(s13, C_LIGHT_BG)
add_header(s13, "Critical Scientific Insights: A Candid, Rigorous Assessment", "Critical Discussion")

# 3 Cards Highlighting Key Truths
insights = [
    ("Statistical Significance (Wilcoxon Test)", 
     "• Paired Wilcoxon Signed-Rank Test: p = 0.3125 (> 0.05).\n"
     "• Fold standard deviations (0.02 - 0.12) are larger than inter-method differences.\n"
     "• Scientific Conclusion: We make NO claim of statistically significant superiority over SMOTE. The methods are competitive, not universally superior.", 
     RGBColor(254, 242, 242), C_ACCENT_WARN),
    
    ("Complexity & Wall-Clock Overhead",
     "• Fitting Base Random Forest: O(T * p * N log N)\n"
     "• Exact TreeSHAP computation: O(N * T * L * D²)\n"
     "• Wall-Clock Cost: Adds ~1.6 seconds per 5-fold CV run.\n"
     "• Asymptotically bounded because attribution is calculated only once per training fold, independent of generated count N_gen.",
     RGBColor(238, 242, 255), C_PRIMARY),
    
    ("Transparent Bug Reporting",
     "• During XGBoost evaluation, TreeSHAP encountered a dimension exception caught by a silent handler, producing tau = 0.0.\n"
     "• Instead of hiding or interpolating data, our paper explicitly excluded corrupt rows and published the diagnostic.\n"
     "• Demonstrates uncompromising academic rigor and research integrity.",
     RGBColor(254, 243, 199), C_ACCENT_GOLD),
]

for idx, (title, text, bg, border) in enumerate(insights):
    left = Inches(0.8 + idx * 3.95)
    add_card(s13, left, Inches(1.6), Inches(3.8), Inches(5.3), bg_color=bg, border_color=border)
    b = s13.shapes.add_textbox(left + Inches(0.2), Inches(1.8), Inches(3.4), Inches(4.9))
    tf = b.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = title
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = border
    p1.space_after = Pt(12)
    p2 = tf.add_paragraph()
    p2.text = text
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_TEXT_DARK

add_speaker_note(s13,
    "What sets our paper apart from typical conference papers is rigorous scientific honesty. "
    "We conducted paired Wilcoxon signed-rank tests which showed p = 0.3125, meaning we cannot claim statistically significant superiority. "
    "We also transparently reported an implementation limitation in the XGBoost attribution handler and documented its computational complexity."
)

# ==========================================================
# SLIDE 14: Limitations & Future Research Roadmap
# ==========================================================
s14 = prs.slides.add_slide(blank_layout)
set_slide_background(s14, C_LIGHT_BG)
add_header(s14, "Limitations & Future Research Roadmap", "Future Directions")

add_card(s14, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=C_CARD_BG, border_color=C_BORDER)
b_l = s14.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.9))
tf_l = b_l.text_frame
tf_l.word_wrap = True

p = tf_l.paragraphs[0]
p.text = "Current Limitations of the Study"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_ACCENT_WARN
p.space_after = Pt(8)

lims = [
    "Simulated Benchmarks: 2 of the 3 primary benchmark datasets are synthetic/simulated; conclusions on real production data remain preliminary.",
    "Numeric-Only Spaces: Evaluated strictly on continuous numerical attributes. Did not address categorical encoding or mixed HEOM distance metrics.",
    "Fixed Decision Threshold (0.5): Did not incorporate cost-sensitive threshold tuning or post-hoc Platt / Isotonic probability recalibration.",
    "Hyperparameter Tuning: Fixed baseline parameters were used without nested grid-search tuning."
]
for lim in lims:
    pl = tf_l.add_paragraph()
    pl.text = "• " + lim
    pl.font.size = Pt(11)
    pl.font.color.rgb = C_TEXT_DARK
    pl.space_after = Pt(8)

add_card(s14, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=RGBColor(240, 253, 244), border_color=C_SECONDARY)
b_r = s14.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
tf_r = b_r.text_frame
tf_r.word_wrap = True

p = tf_r.paragraphs[0]
p.text = "Future Research Directions"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_SECONDARY
p.space_after = Pt(8)

futs = [
    "Touchpoint T4 (Post-Generation Validation): Incorporate generate-and-test candidate filtering to enforce Kendall rank preservation constraints.",
    "Mixed Data Types: Extend metric tensor W to Gower / HEOM distance metrics for mixed categorical and numerical tables.",
    "Probability Recalibration: Combine oversampling with calibrated classifiers to fix ECE distortion in clinical deployment.",
    "Expanded Real-World Benchmarks: Evaluate against 20+ real OpenML tabular datasets with automated hyperparameter optimization."
]
for fut in futs:
    pf = tf_r.add_paragraph()
    pf.text = "• " + fut
    pf.font.size = Pt(11)
    pf.font.color.rgb = C_TEXT_DARK
    pf.space_after = Pt(8)

add_speaker_note(s14,
    "We outline key limitations and an actionable future roadmap. "
    "Future work will extend our framework to mixed categorical-numerical datasets, integrate post-generation validation touchpoints (T4), "
    "and apply probability recalibration methods to ensure reliable risk assessment in healthcare."
)

# ==========================================================
# SLIDE 15: Conclusion & Summary
# ==========================================================
s15 = prs.slides.add_slide(blank_layout)
set_slide_background(s15, C_DARK_BG)

# Title container
tbox = s15.shapes.add_textbox(Inches(1.0), Inches(1.0), Inches(11.3), Inches(1.2))
tf = tbox.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Summary of Contributions & Final Conclusion"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(255, 255, 255)

# 4 Key Takeaways Cards
takeaways = [
    ("1. Novel Upstream Paradigm", "Pioneered shifting TreeSHAP from passive post-hoc model inspection to active upstream synthetic sample generation.", RGBColor(30, 41, 59)),
    ("2. Two Modular Formulations", "Designed SHAP-Weighted SMOTE (metric deformation via coordinate warping) and SHAP-Filtered SMOTE (subspace pruning).", RGBColor(30, 41, 59)),
    ("3. Zero-Data-Leakage Standard", "Demonstrated how global resampling falsely inflates PR-AUC by >100%, and established a strict in-fold benchmarking standard.", RGBColor(30, 41, 59)),
    ("4. Honest Empirical Evidence", "Proved that while SHAP-Weighted boosts minority F1 in overlapping spaces, it is competitive rather than universally superior.", RGBColor(30, 41, 59)),
]

for idx, (title, desc, bg) in enumerate(takeaways):
    left = Inches(1.0 + (idx % 2) * 5.75)
    top = Inches(2.3 + (idx // 2) * 2.1)
    add_card(s15, left, top, Inches(5.6), Inches(1.9), bg_color=bg, border_color=RGBColor(59, 130, 246))
    b = s15.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), Inches(5.2), Inches(1.5))
    tf = b.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = title
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(96, 165, 250)
    p1.space_after = Pt(4)
    p2 = tf.add_paragraph()
    p2.text = desc
    p2.font.size = Pt(11)
    p2.font.color.rgb = RGBColor(226, 232, 240)

# Footer repo link
fbox = s15.shapes.add_textbox(Inches(1.0), Inches(6.4), Inches(11.3), Inches(0.6))
tf_f = fbox.text_frame
tf_f.word_wrap = True
pf = tf_f.paragraphs[0]
pf.alignment = PP_ALIGN.CENTER
pf.text = "Open Source Code & Data: https://github.com/hasanr-26/SHAP-SMOTE  •  Thank You! Questions?"
pf.font.size = Pt(13)
pf.font.bold = True
pf.font.color.rgb = RGBColor(255, 255, 255)

add_speaker_note(s15,
    "To conclude, our study provides both an innovative algorithmic approach to explainability-guided oversampling "
    "and a critical, leakage-free empirical benchmark. We invite you to explore our open-source codebase on GitHub. "
    "Thank you for your time, and we would be happy to answer any questions."
)

# Save presentation
output_path = os.path.join(REPO_ROOT, "Explainability_Aware_Oversampling_Presentation.pptx")
prs.save(output_path)
print(f"Presentation saved successfully to: {output_path}")
