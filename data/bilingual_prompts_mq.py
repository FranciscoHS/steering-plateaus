"""
30 medium-quality semantically matched English-Mandarin prompt pairs for language steering experiments.

More encyclopedic/mechanical tone than the HQ set. Structural matching is decent but not as tight.
Some patterns are repetitive ("The X is Y" style). Used as middle tier in feature purity experiments.
"""

PROMPT_PAIRS = [
    # Science & Nature (1-5)
    ("The way photosynthesis works is", "光合作用的原理是"),
    ("When water freezes, it", "水结冰时会"),
    ("Because gravity affects all objects,", "由于重力作用于所有物体，"),
    ("If we study the layers of the earth,", "如果我们研究地球的各层，"),
    ("Enzymes speed up chemical reactions by", "酶通过加快化学反应"),

    # Technology (6-10)
    ("The main advantage of artificial intelligence is", "人工智能的主要优势是"),
    ("Blockchain technology became important because", "区块链技术变得重要是因为"),
    ("After data is encrypted, it", "数据被加密后会"),
    ("Smartphones have changed how people", "智能手机改变了人们"),
    ("Programming languages differ in", "编程语言的差异在于"),

    # Food & Cooking (11-15)
    ("Fermentation preserves food by", "发酵通过保存食物"),
    ("When you cook meat at high temperatures,", "当你在高温下烹饪肉类时，"),
    ("The reason salt enhances flavor is", "盐增强风味的原因是"),
    ("If you want to bake bread successfully,", "如果你想成功烤面包，"),
    ("Olive oil differs from vegetable oil in", "橄榄油与植物油的不同之处在于"),

    # History & Culture (16-20)
    ("Ancient civilizations developed writing systems to", "古代文明发展书写系统是为了"),
    ("After the invention of the printing press,", "印刷术发明后，"),
    ("The reason trade routes were established was", "建立贸易路线的原因是"),
    ("Throughout human history, languages have", "纵观人类历史，语言"),
    ("Because of cultural exchange, different regions", "由于文化交流，不同地区"),

    # Health & Biology (21-25)
    ("Regular exercise strengthens the body by", "定期锻炼通过强化身体"),
    ("When the immune system encounters a virus,", "当免疫系统遇到病毒时，"),
    ("Sleep is essential for humans because", "睡眠对人类至关重要是因为"),
    ("If you maintain a balanced diet, you", "如果你保持均衡饮食，你会"),
    ("The main difference between bacteria and viruses is", "细菌和病毒的主要区别是"),

    # Weather & Environment (26-30)
    ("Clouds form when moisture in the air", "当空气中的水分时，云会形成"),
    ("After it rains heavily, the soil", "大雨过后，土壤会"),
    ("Wind patterns influence ocean currents by", "风型通过影响洋流"),
    ("Because plants need sunlight to grow,", "因为植物需要阳光才能生长，"),
    ("Temperature affects the speed of chemical reactions,", "温度影响化学反应的速度，"),
]

# Convenience accessors
ENGLISH_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
CHINESE_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
