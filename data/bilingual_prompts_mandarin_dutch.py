"""
50 semantically matched Mandarin-Dutch prompt pairs for language steering experiments.

Each pair expresses the same meaning, designed to elicit natural continuations
in the respective language. Token lengths are kept similar (~3-6 tokens per side).
"""

PROMPT_PAIRS = [
    # Science & Nature
    ("太阳很温暖", "De zon is warm"),
    ("水非常重要", "Water is erg belangrijk"),
    ("今天的天气是", "Het weer vandaag is"),
    ("这棵树很大", "De boom is groot"),
    ("大海非常", "De zee is heel"),
    ("地球围绕太阳", "De aarde draait om"),
    ("星星在夜晚闪耀", "Sterren schijnen 's nachts"),
    ("山上有很多", "De berg heeft veel"),
    ("河水流向大海的", "Het water stroomt naar"),
    ("花园里的花", "De bloemen in de"),

    # Daily Life
    ("我喜欢读书", "Ik lees heel graag"),
    ("她去了学校", "Zij gaat naar school"),
    ("我们住在一个", "Wij wonen in een"),
    ("他在这里工作", "Hij werkt hier bij"),
    ("孩子们在外面", "De kinderen spelen buiten"),
    ("我每天早上", "Ik sta elke ochtend"),
    ("他们一起走路", "Zij lopen samen door"),
    ("我的朋友很好", "Mijn vriend is goed"),
    ("她打开了门", "Zij opende de deur"),
    ("我们需要更多", "Wij hebben veel meer"),

    # Food & Drink
    ("我总是喝咖啡", "Ik drink altijd koffie"),
    ("这里的食物很", "Het eten hier is"),
    ("我们一起吃饭", "Wij eten samen om"),
    ("面包很新鲜", "Het brood is vers"),
    ("水果非常好吃", "Het fruit is lekker"),

    # Places & Travel
    ("这个城市有很多", "De stad heeft veel"),
    ("火车晚了", "De trein is laat"),
    ("我的房子很大", "Mijn huis is groot"),
    ("我来自一个", "Ik kom uit een"),
    ("这个国家很美", "Dit land is mooi"),

    # Knowledge & Education
    ("学校今天开门", "De school is open"),
    ("学生们学习很努力", "De studenten studeren hard"),
    ("老师解释了", "De leraar legt uit"),
    ("读书可以提高", "Lezen kan je verbeteren"),
    ("教育很重要因为", "Onderwijs is belangrijk"),

    # Family & Home
    ("我的母亲做饭很好", "Mijn moeder kookt goed"),
    ("宝宝现在在睡觉", "De baby slaapt nu"),
    ("我们的花园很大", "Onze tuin is groot"),
    ("我的父亲在这里", "Mijn vader werkt hier"),
    ("家人一起吃晚饭", "Het gezin eet samen"),

    # Health
    ("医生说", "De dokter zegt dat"),
    ("我感觉很好", "Ik voel me goed"),
    ("充足的睡眠对于身体", "Slaap is erg belangrijk"),
    ("运动对健康有益", "Bewegen is goed voor"),
    ("医院就在附近的", "Het ziekenhuis is dichtbij"),

    # Weather & Seasons
    ("雨下得很大", "De regen valt hard"),
    ("夏天是我最喜欢的", "Zomer is mijn favoriet"),
    ("风吹得很强", "De wind waait sterk"),
    ("春天在冬天之后", "Lente komt na winter"),
    ("雪是白色的", "De sneeuw is wit"),
]

# Convenience accessors
CHINESE_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
DUTCH_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
