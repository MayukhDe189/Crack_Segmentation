import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_pptx="slides/crack_segmentation_presentation.pptx"):
    os.makedirs(os.path.dirname(output_pptx), exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Theme colors
    NAVY = RGBColor(24, 43, 73)
    SLATE = RGBColor(50, 70, 95)
    LIGHT_BG = RGBColor(245, 247, 250)
    DARK_TEXT = RGBColor(30, 30, 30)
    WHITE = RGBColor(255, 255, 255)
    ACCENT_BLUE = RGBColor(31, 119, 180)
    ACCENT_RED = RGBColor(214, 39, 40)
    ACCENT_GREEN = RGBColor(44, 160, 44)
    GRAY_BORDER = RGBColor(210, 215, 220)

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="MACHINE LEARNING PROJECT • IMAGE SEGMENTATION"):
        # Top banner background
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.15))
        shape.fill.solid()
        shape.fill.fore_color.rgb = NAVY
        shape.line.fill.background()

        # Category Tracker
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.7), Inches(0.3))
        p_cat = tb_cat.text_frame.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = RGBColor(160, 195, 240)

        # Main Title
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.7))
        p = tb.text_frame.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = WHITE

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    # Title box
    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(3.0))
    p1 = tb1.text_frame.paragraphs[0]
    p1.text = "Pavement Crack Segmentation with U-Net"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p2 = tb1.text_frame.add_paragraph()
    p2.text = "Tackling Extreme Class Imbalance via Validation-Tuned Weighted Cross-Entropy Loss"
    p2.font.size = Pt(20)
    p2.font.color.rgb = RGBColor(180, 210, 245)
    p2.space_before = Pt(14)

    # Info card
    card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.5), Inches(11.333), Inches(1.8))
    card.fill.solid()
    card.fill.fore_color.rgb = RGBColor(34, 58, 96)
    card.line.color.rgb = RGBColor(60, 95, 150)
    
    tb_card = s1.shapes.add_textbox(Inches(1.3), Inches(4.7), Inches(10.7), Inches(1.4))
    tf_c = tb_card.text_frame
    tf_c.word_wrap = True
    
    p_c1 = tf_c.paragraphs[0]
    p_c1.text = "• Problem Statement: Segment thin pavement cracks under heavy background dominance (60:1 ratio)."
    p_c1.font.size = Pt(14)
    p_c1.font.color.rgb = WHITE
    
    p_c2 = tf_c.add_paragraph()
    p_c2.text = "• Exploration & Contribution: Validation tuning of pos-weight (w) in Weighted BCE & Ablation back to w=1.0."
    p_c2.font.size = Pt(14)
    p_c2.font.color.rgb = WHITE
    p_c2.space_before = Pt(6)
    
    p_c3 = tf_c.add_paragraph()
    p_c3.text = "• Dataset: CrackForest Dataset (CFD) • Benchmark Model: Full Encoder-Decoder U-Net (256x256)."
    p_c3.font.size = Pt(14)
    p_c3.font.color.rgb = RGBColor(220, 230, 245)
    p_c3.space_before = Pt(6)

    # =========================================================================
    # SLIDE 2: MOTIVATION & PROBLEM CLARITY
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Problem Motivation & The Class Imbalance Dilemma")

    # Left Column: The Real-World Challenge
    box_l = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    box_l.fill.solid()
    box_l.fill.fore_color.rgb = WHITE
    box_l.line.color.rgb = GRAY_BORDER

    tb_l = s2.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(4.9))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "The Real-World Challenge"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = NAVY

    points_l = [
        ("Automated Road Inspection:", "Timely identification of asphalt cracks prevents catastrophic structural failures, potholes, and costly highway rehabilitation."),
        ("Thin Morphologies:", "Pavement cracks span only 1 to 5 pixels in width across high-resolution pavement photography."),
        ("Complex Textures:", "Rough asphalt surfaces, oil stains, surface aggregate, and shadows present strong noise artifacts that mimic true cracks."),
        ("Safety & Maintenance:", "Reliable automated segmentation enables scalable, continuous monitoring of municipal and civil infrastructure.")
    ]
    for title, desc in points_l:
        p = tf_l.add_paragraph()
        p.text = f"• {title} {desc}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_TEXT
        p.space_before = Pt(10)

    # Right Column: The Core Flaw of Ordinary BCE
    box_r = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    box_r.fill.solid()
    box_r.fill.fore_color.rgb = WHITE
    box_r.line.color.rgb = GRAY_BORDER

    tb_r = s2.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(4.9))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Why Standard Cross-Entropy Fails"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACCENT_RED

    points_r = [
        ("Severe 60:1 Imbalance:", "In CFD, cracks occupy only 1.63% of the image; 98.37% is pavement background."),
        ("Gradient Overwhelming:", "Standard BCE treats all pixels equally. 98% of the backward gradients originate from pavement, driving predictions to 0."),
        ("Fragmented / Broken Lines:", "The baseline model becomes overly conservative. Fine crack branches are dropped, leading to dismal recall (~37.8%)."),
        ("Proposed Remedy:", "Weighted Binary Cross-Entropy introduces a positive weight w > 1 that scales gradients on crack pixels, recovering continuous boundaries.")
    ]
    for title, desc in points_r:
        p = tf_r.add_paragraph()
        p.text = f"• {title} {desc}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_TEXT
        p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 3: LITERATURE REVIEW
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. Literature Review: Evolution of Crack Detection Methods")

    cards_data = [
        ("Traditional Thresholding & Edge Filters", 
         "• Methods: Otsu thresholding, Canny, Sobel edge detectors, Morphological filters.\n"
         "• Strengths: Fast, rule-based, zero training required.\n"
         "• Severe Limitations: Highly sensitive to uneven lighting, tire marks, shadows, and aggregate textures. Fails completely in unconstrained road scenes."),
        ("Random Forests & Machine Learning", 
         "• Methods: CrackTree (Zou et al.), Random Structured Forests (Shi & Cui et al., 2016).\n"
         "• Strengths: Learns local texture and edge patterns on hand-engineered features.\n"
         "• Limitations: Requires manual feature engineering; unable to model long-range topological crack continuity."),
        ("Deep FCNs & Standard U-Net", 
         "• Methods: Fully Convolutional Networks (Long et al.), Standard U-Net (Ronneberger et al.).\n"
         "• Strengths: End-to-end multi-scale representation with skip connections preserving spatial resolution.\n"
         "• Primary Flaw: When trained with unweighted BCE, the loss is dominated by background pixels, producing disconnected, broken cracks.")
    ]

    for i, (head, body) in enumerate(cards_data):
        y_pos = Inches(1.5 + i * 1.85)
        cd = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y_pos, Inches(11.7), Inches(1.65))
        cd.fill.solid()
        cd.fill.fore_color.rgb = WHITE
        cd.line.color.rgb = GRAY_BORDER

        tb = s3.shapes.add_textbox(Inches(1.0), y_pos + Inches(0.12), Inches(11.3), Inches(1.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = head
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = NAVY

        for line in body.split("\n"):
            p2 = tf.add_paragraph()
            p2.text = line
            p2.font.size = Pt(12)
            p2.font.color.rgb = DARK_TEXT
            p2.space_before = Pt(3)

    # =========================================================================
    # SLIDE 4: DATASET & PREPROCESSING PIPELINE
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Dataset Curation, Splits & Preprocessing Pipeline")

    # Left Box: CrackForest Dataset (CFD)
    b4_1 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    b4_1.fill.solid()
    b4_1.fill.fore_color.rgb = WHITE
    b4_1.line.color.rgb = GRAY_BORDER

    tb4_1 = s4.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(4.9))
    tf4_1 = tb4_1.text_frame
    tf4_1.word_wrap = True
    p = tf4_1.paragraphs[0]
    p.text = "CrackForest Dataset (CFD) Characteristics"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = NAVY

    cfd_points = [
        "Origin: Cui & Shi et al., Institute of Remote Sensing and Digital Earth.",
        "Images: 118 high-resolution colour pavement images (original ~480x320).",
        "Ground Truth: Annotated MATLAB .mat structures containing binary crack segmentation matrices.",
        "Total Crack Pixels: 292,885 (1.63% foreground).",
        "Total Background Pixels: 17,710,362 (98.37% background).",
        "True Class Imbalance: 60.47 : 1 pavement-to-crack ratio."
    ]
    for pt in cfd_points:
        p = tf4_1.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_TEXT
        p.space_before = Pt(8)

    # Right Box: Preprocessing & Rigorous Splits
    b4_2 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    b4_2.fill.solid()
    b4_2.fill.fore_color.rgb = WHITE
    b4_2.line.color.rgb = GRAY_BORDER

    tb4_2 = s4.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(4.9))
    tf4_2 = tb4_2.text_frame
    tf4_2.word_wrap = True
    p = tf4_2.paragraphs[0]
    p.text = "Data Splits & Preprocessing Protocol"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    prep_points = [
        "Target Resolution: Resized to 256 x 256 (Image: Bilinear, Mask: Nearest Neighbor to prevent boundary blurring).",
        "Deterministic Splits (Seed = 42):",
        "   - Training Set: 82 images (69.5%)",
        "   - Validation Set: 17 images (14.4%) — used for weight tuning",
        "   - Test Set: 19 images (16.1%) — held out for final evaluation",
        "Exact Alignment: Both baseline (w=1) and proposed (w=w*) models train on identical splits.",
        "Augmentations: Random horizontal & vertical flips, ImageNet channel-wise normalization.",
        "In-Memory Cache: All 118 samples pre-cached in RAM for zero I/O overhead."
    ]
    for pt in prep_points:
        p = tf4_2.add_paragraph()
        p.text = f"• {pt}" if not pt.startswith("   ") else pt
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_TEXT
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 5: U-NET ARCHITECTURE
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Deep Architecture: Standard U-Net for Binary Segmentation")

    # Architecture Overview Left
    b5_1 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(6.5), Inches(5.3))
    b5_1.fill.solid()
    b5_1.fill.fore_color.rgb = WHITE
    b5_1.line.color.rgb = GRAY_BORDER

    tb5_1 = s5.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(6.1), Inches(4.9))
    tf5_1 = tb5_1.text_frame
    tf5_1.word_wrap = True
    p = tf5_1.paragraphs[0]
    p.text = "U-Net Structural Specifications"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = NAVY

    unet_specs = [
        ("Encoder Stage:", "4 downsampling blocks with DoubleConv [3x3 Conv -> BatchNorm -> ReLU -> 3x3 Conv -> BatchNorm -> ReLU] followed by 2x2 Max Pooling. Channels: 3 -> 16 -> 32 -> 64 -> 128."),
        ("Bottleneck:", "DoubleConv expanding to 256 channels, capturing high-level contextual receptive fields."),
        ("Decoder Stage:", "4 upsampling blocks via ConvTranspose2d (stride 2) concatenating corresponding encoder skip connections. Channels: 256 -> 128 -> 64 -> 32 -> 16."),
        ("Skip Connections:", "Crucial for crack segmentation: bypasses low-level boundary details directly to decoder, preventing thin lines from vanishing during downsampling."),
        ("Classification Head:", "1x1 Conv mapping 16 channels to 1 logit output channel followed by Sigmoid activation.")
    ]
    for title, desc in unet_specs:
        p = tf5_1.add_paragraph()
        p.text = f"• {title} {desc}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_TEXT
        p.space_before = Pt(6)

    # Right Box: Hyperparameter Configuration
    b5_2 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.7), Inches(1.5), Inches(4.8), Inches(5.3))
    b5_2.fill.solid()
    b5_2.fill.fore_color.rgb = WHITE
    b5_2.line.color.rgb = GRAY_BORDER

    tb5_2 = s5.shapes.add_textbox(Inches(7.9), Inches(1.7), Inches(4.4), Inches(4.9))
    tf5_2 = tb5_2.text_frame
    tf5_2.word_wrap = True
    p = tf5_2.paragraphs[0]
    p.text = "Model & Training Hyperparameters"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    hp_table = [
        ("Parameter", "Value"),
        ("Input Dimensions", "256 x 256 x 3 (RGB)"),
        ("Output Dimensions", "256 x 256 x 1 (Logits)"),
        ("Total Parameters", "1,942,577 (1.94 Million)"),
        ("Optimizer", "Adam (lr = 1.5e-3, decay = 1e-5)"),
        ("LR Scheduler", "CosineAnnealingLR (T_max = 20)"),
        ("Batch Size", "8 samples per batch"),
        ("Training Epochs", "20 epochs per model"),
        ("Hardware / Threads", "Multi-threaded CPU (16 threads)")
    ]
    for k, v in hp_table[1:]:
        p = tf5_2.add_paragraph()
        p.text = f"• {k}: {v}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_TEXT
        p.space_before = Pt(7)

    # =========================================================================
    # SLIDE 6: PROPOSED METHODOLOGY & MATHEMATICAL FORMULATION
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. Proposed Methodology: Weighted Binary Cross-Entropy (WBCE)")

    # Left: Math Formulation
    b6_1 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.3))
    b6_1.fill.solid()
    b6_1.fill.fore_color.rgb = WHITE
    b6_1.line.color.rgb = GRAY_BORDER

    tb6_1 = s6.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(4.9))
    tf6_1 = tb6_1.text_frame
    tf6_1.word_wrap = True
    p = tf6_1.paragraphs[0]
    p.text = "Mathematical Loss Formulation"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p = tf6_1.add_paragraph()
    p.text = "1. Ordinary Binary Cross-Entropy (Baseline, w = 1.0):"
    p.font.size = Pt(13)
    p.font.bold = True
    p.space_before = Pt(8)

    p = tf6_1.add_paragraph()
    p.text = "   L_BCE = - (1/N) * sum [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]"
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(10, 80, 160)
    p.space_before = Pt(4)

    p = tf6_1.add_paragraph()
    p.text = "2. Weighted Binary Cross-Entropy (Proposed, w > 1.0):"
    p.font.size = Pt(13)
    p.font.bold = True
    p.space_before = Pt(10)

    p = tf6_1.add_paragraph()
    p.text = "   L_WBCE(w) = - (1/N) * sum [ w * y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]"
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_RED
    p.space_before = Pt(4)

    p = tf6_1.add_paragraph()
    p.text = "Where y_i in {0, 1} is crack ground truth, p_i = sigmoid(z_i) is probability, and w is the positive class weighting factor."
    p.font.size = Pt(11)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(8)

    # Right: The Ablation Principle
    b6_2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.3))
    b6_2.fill.solid()
    b6_2.fill.fore_color.rgb = WHITE
    b6_2.line.color.rgb = GRAY_BORDER

    tb6_2 = s6.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.2), Inches(4.9))
    tf6_2 = tb6_2.text_frame
    tf6_2.word_wrap = True
    p = tf6_2.paragraphs[0]
    p.text = "Theoretical Rationale & Ablation Principle"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    p = tf6_2.add_paragraph()
    p.text = "• Gradient Scaling Effect:"
    p.font.size = Pt(13)
    p.font.bold = True
    p.space_before = Pt(8)

    p = tf6_2.add_paragraph()
    p.text = "   d(L_WBCE) / dz_i = p_i - y_i * (1 + (w - 1) * (1 - p_i)). When a crack pixel is falsely predicted as background (p_i ≈ 0), the gradient magnitude is multiplied by w, forcing intense backpropagation."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(3)

    p = tf6_2.add_paragraph()
    p.text = "• The Controlled Ablation (w = 1.0):"
    p.font.size = Pt(13)
    p.font.bold = True
    p.space_before = Pt(12)

    p = tf6_2.add_paragraph()
    p.text = "   When w is set back to 1.0, the positive penalty term collapses back to standard unweighted BCE. Comparing w = 1.0 against the chosen w* isolates the exact performance contribution of the class reweighting strategy."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(3)

    # =========================================================================
    # SLIDE 7: HYPERPARAMETER TUNING & VALIDATION SPLIT SELECTION
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. Validation Split Tuning: Choosing the Optimal Weight (w*)")

    # Embed validation curve image
    val_img_path = "results/validation_weight_tuning.png"
    if os.path.exists(val_img_path):
        s7.shapes.add_picture(val_img_path, Inches(0.8), Inches(1.5), width=Inches(6.6))

    # Right: Summary Table & Analysis
    b7 = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.7), Inches(1.5), Inches(4.8), Inches(5.3))
    b7.fill.solid()
    b7.fill.fore_color.rgb = WHITE
    b7.line.color.rgb = GRAY_BORDER

    tb7 = s7.shapes.add_textbox(Inches(7.9), Inches(1.7), Inches(4.4), Inches(4.9))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    p = tf7.paragraphs[0]
    p.text = "Validation Tuning Results (w)"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = NAVY

    t_data = [
        ("w = 1.0 (Baseline)", "Val F1: 0.5411 | IoU: 0.3709 | Recall: 45.3%"),
        ("w = 5.0 (Optimal w*)", "Val F1: 0.6153 | IoU: 0.4444 | Recall: 79.7%"),
        ("w = 10.0", "Val F1: 0.5803 | IoU: 0.4088 | Recall: 83.5%"),
        ("w = 15.0", "Val F1: 0.5456 | IoU: 0.3751 | Recall: 90.1%")
    ]
    for w_name, metrics in t_data:
        p = tf7.add_paragraph()
        p.text = f"• {w_name}:"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_RED if "Optimal" in w_name else DARK_TEXT
        p.space_before = Pt(8)
        
        p_sub = tf7.add_paragraph()
        p_sub.text = f"   {metrics}"
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = DARK_TEXT

    p_conc = tf7.add_paragraph()
    p_conc.text = "Key Insight: As w increases, Recall increases monotonically (45.3% -> 90.1%). w* = 5.0 strikes the optimal harmonic balance between precision and recall, achieving the peak Validation F1 (0.6153) and IoU (0.4444)."
    p_conc.font.size = Pt(11)
    p_conc.font.color.rgb = NAVY
    p_conc.space_before = Pt(12)

    # =========================================================================
    # SLIDE 8: TEST RESULTS & COMPARATIVE ANALYSIS
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "7. Quantitative Results: Rigorous Test Split Evaluation")

    # Embed Bar Chart
    bar_img_path = "results/metrics_comparison_bar.png"
    if os.path.exists(bar_img_path):
        s8.shapes.add_picture(bar_img_path, Inches(0.8), Inches(1.5), width=Inches(6.4))

    # Right Box: Comparison Table & Deltas
    b8 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.5), Inches(1.5), Inches(5.0), Inches(5.3))
    b8.fill.solid()
    b8.fill.fore_color.rgb = WHITE
    b8.line.color.rgb = GRAY_BORDER

    tb8 = s8.shapes.add_textbox(Inches(7.7), Inches(1.7), Inches(4.6), Inches(4.9))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    p = tf8.paragraphs[0]
    p.text = "Held-Out Test Set Performance"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = NAVY

    res_summary = [
        ("Metric", "Baseline (w=1)", "Proposed (w=5)", "Delta"),
        ("Recall", "0.3784", "0.7809", "+106.4%"),
        ("F1 Score", "0.4862", "0.5958", "+22.6%"),
        ("IoU (Jaccard)", "0.3211", "0.4243", "+32.1%"),
        ("Precision", "0.6797", "0.4817", "-0.1980")
    ]
    for row in res_summary[1:]:
        p = tf8.add_paragraph()
        p.text = f"• {row[0]}: {row[1]} -> {row[2]}  ({row[3]})"
        p.font.size = Pt(13)
        p.font.bold = True if row[0] in ["Recall", "F1 Score", "IoU (Jaccard)"] else False
        p.font.color.rgb = ACCENT_GREEN if "+" in row[3] else DARK_TEXT
        p.space_before = Pt(8)

    p_takeaway = tf8.add_paragraph()
    p_takeaway.text = "Major Takeaway: The baseline model missed 62.2% of all crack pixels (Recall = 37.8%). The proposed weighted model captured 78.1% of all cracks (+106.4% recall boost), resulting in a +22.6% higher F1 score and +32.1% higher IoU on completely unseen pavement images."
    p_takeaway.font.size = Pt(11)
    p_takeaway.font.color.rgb = NAVY
    p_takeaway.space_before = Pt(14)

    # =========================================================================
    # SLIDE 9: QUALITATIVE VISUAL ANALYSIS
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "8. Qualitative Mask Interpretation: Crack Continuity vs. Fragmentation")

    # Embed Test Predictions Comparison Grid
    pred_img_path = "results/test_predictions_comparison.png"
    if os.path.exists(pred_img_path):
        s9.shapes.add_picture(pred_img_path, Inches(0.8), Inches(1.4), width=Inches(8.5))

    # Right Card: Visual Interpretation Points
    b9 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.6), Inches(1.4), Inches(2.9), Inches(5.6))
    b9.fill.solid()
    b9.fill.fore_color.rgb = WHITE
    b9.line.color.rgb = GRAY_BORDER

    tb9 = s9.shapes.add_textbox(Inches(9.75), Inches(1.5), Inches(2.6), Inches(5.3))
    tf9 = tb9.text_frame
    tf9.word_wrap = True
    p = tf9.paragraphs[0]
    p.text = "Visual Inspection"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY

    interp_points = [
        ("Baseline (w=1.0):", "Severe fragmentation. Only thick crack cores are flagged; faint lateral branches are dropped completely."),
        ("Proposed (w=5.0):", "Preserves continuous topological crack traces without breaking them apart."),
        ("Overlay View:", "Red predictions tightly follow the true road fissure path with sharp boundary localization."),
        ("Ablation Conclusion:", "Removing the weight (setting w=1) immediately breaks crack continuity.")
    ]
    for h, b in interp_points:
        p = tf9.add_paragraph()
        p.text = f"{h} {b}"
        p.font.size = Pt(10)
        p.font.color.rgb = DARK_TEXT
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 10: CONCLUSION & VIVA PREPARATION
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "9. Conclusion & Technical Viva Preparation Guide")

    # Three takeaway cards across the slide
    card_w = Inches(3.7)
    card_h = Inches(5.3)
    y_pos = Inches(1.5)

    c_data = [
        ("Core Findings", NAVY, [
            "Severe 60:1 Imbalance: Thin cracks are drowned out under standard BCE.",
            "Weight Tuning: On the validation split, w* = 5.0 achieved the peak F1 score (0.6153).",
            "Generalization: On unseen test pavement images, recall jumped from 37.8% to 78.1% (+106.4%).",
            "Overall F1 increased from 0.4862 to 0.5958 (+22.6%)."
        ]),
        ("Technical Viva Defense", ACCENT_BLUE, [
            "Q: Why did baseline produce broken cracks?",
            "A: 98.4% background pixels dominated backward gradients, biasing model output toward 0.",
            "Q: Why not use w = 50 or w = 60?",
            "A: Excessive w leads to high false positives, marking asphalt texture as cracks. Tuning on validation split prevents over-prediction.",
            "Q: What is the ablation?",
            "A: Setting w=1 exactly recovers ordinary BCE, proving the weight is the decisive factor."
        ]),
        ("Future Research Directions", ACCENT_RED, [
            "Compound Loss Objectives: Combine Weighted BCE with Soft Dice Loss or Focal Loss.",
            "Attention U-Net / Transformers: Incorporate spatial attention gates to suppress non-crack asphalt background noise.",
            "Multi-Scale Feature Aggregation: Dilated convolutions or ASPP for wide fissure detection.",
            "Edge Computing: Quantization for real-time UAV and road inspection vehicle cameras."
        ])
    ]

    for i, (head, col, bullets) in enumerate(c_data):
        x_pos = Inches(0.8 + i * 4.0)
        card = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x_pos, y_pos, card_w, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = GRAY_BORDER

        tb = s10.shapes.add_textbox(x_pos + Inches(0.15), y_pos + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = head
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = col

        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(11)
            p.font.color.rgb = DARK_TEXT
            p.space_before = Pt(7)

    prs.save(output_pptx)
    print(f"Presentation saved successfully to: {output_pptx}")

if __name__ == "__main__":
    create_deck()
