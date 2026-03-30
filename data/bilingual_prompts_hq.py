"""
29 high-quality semantically matched English-Mandarin prompt pairs for language steering experiments.

Each pair expresses the same meaning with matched grammatical structure (conditional↔conditional,
causal↔causal, temporal↔temporal, etc.), designed to elicit natural continuations in the
respective language. Structural matching is prioritized to produce clean contrastive directions.
"""

PROMPT_PAIRS = [
    # --- Science ---
    ("Because the experiment failed,", "因为实验失败了，"),
    ("The speed of light in a medium", "光在介质中的速度"),
    ("When two atoms collide,", "当两个原子碰撞时，"),

    # --- Daily life ---
    ("After waking up this morning,", "今天早上醒来后，"),
    ("The easiest way to fix this is", "最简单的修复方法是"),
    ("I usually prefer to", "我通常更喜欢"),

    # --- History ---
    ("Ancient civilizations often", "古代文明常常"),
    ("Although the war had ended,", "虽然战争已经结束，"),
    ("Long before writing was invented,", "在文字发明之前很久，"),

    # --- Technology ---
    ("The latest software update", "最新的软件更新"),
    ("In order to improve efficiency,", "为了提高效率，"),
    ("If the system crashes,", "如果系统崩溃了，"),

    # --- Food ---
    ("If you add too much salt,", "如果盐放太多，"),
    ("Once the water starts boiling,", "水烧开之后，"),
    ("The secret to a good sauce is", "好酱汁的秘诀在于"),

    # --- Health ---
    ("Regular exercise can help", "经常锻炼可以帮助"),
    ("If you don't sleep enough,", "如果睡眠不足，"),
    ("After taking the medicine,", "吃完药之后，"),

    # --- Weather / Nature ---
    ("When the storm finally passes,", "暴风雨过后，"),
    ("Birds migrate south because", "鸟类南迁是因为"),
    ("Since the river dried up,", "自从河流干涸后，"),

    # --- Culture / Philosophy ---
    ("The purpose of this tradition is", "这个传统的目的是"),
    ("Compared to the old method,", "与旧方法相比，"),
    ("People often forget that", "人们常常忘记"),

    # --- Economics ---
    ("If prices keep rising,", "如果价格持续上涨，"),
    ("Ever since the market crashed,", "自从市场崩盘以来，"),

    # --- Education / Psychology ---
    ("The best way to learn is", "最好的学习方法是"),
    ("Human memory tends to", "人的记忆往往会"),

    # --- Art / Music ---
    ("The colors in this painting", "这幅画中的颜色"),
    ("Although the melody sounds simple,", "虽然旋律听起来简单，"),
]

# Convenience accessors
ENGLISH_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
CHINESE_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
