"""
50 semantically matched English-Chinese prompt pairs for language steering experiments.

Each pair expresses the same meaning, designed to elicit natural continuations
in the respective language.
"""

PROMPT_PAIRS = [
    # Science & Nature
    ("The sun rises in the", "太阳从"),
    ("Water boils at a temperature of", "水在温度达到"),
    ("The largest planet in our solar system is", "太阳系中最大的行星是"),
    ("Trees produce oxygen through", "树木通过"),
    ("The speed of light is approximately", "光速大约是"),
    ("Earthquakes are caused by", "地震是由"),
    ("The human brain contains about", "人类大脑大约包含"),
    ("Diamonds are made of", "钻石是由"),
    ("The deepest ocean in the world is", "世界上最深的海洋是"),
    ("Photosynthesis is the process by which", "光合作用是"),

    # Geography & History
    ("The capital of Japan is", "日本的首都是"),
    ("The Great Wall of China was built to", "长城的建造是为了"),
    ("The longest river in Africa is", "非洲最长的河流是"),
    ("The population of India is approximately", "印度的人口大约是"),
    ("Mount Everest is located in", "珠穆朗玛峰位于"),
    ("The ancient Egyptians built pyramids to", "古埃及人建造金字塔是为了"),
    ("The Pacific Ocean is the", "太平洋是"),
    ("The first person to walk on the moon was", "第一个登上月球的人是"),
    ("Australia is known for its", "澳大利亚以其"),
    ("The Sahara Desert covers an area of", "撒哈拉沙漠的面积约为"),

    # Daily Life
    ("Breakfast is the most important meal of", "早餐是一天中最重要的"),
    ("Many people enjoy drinking coffee in the", "很多人喜欢在"),
    ("Exercise is important because it helps", "运动很重要，因为它有助于"),
    ("Reading books can improve your", "读书可以提高你的"),
    ("A balanced diet should include", "均衡的饮食应该包含"),
    ("Getting enough sleep is essential for", "充足的睡眠对于"),
    ("Learning a new language requires", "学习一门新语言需要"),
    ("Cooking at home is healthier than", "在家做饭比"),
    ("Music can help people feel more", "音乐可以帮助人们感到更"),
    ("Traveling to new places helps us", "去新的地方旅行帮助我们"),

    # Technology
    ("Artificial intelligence is changing the way we", "人工智能正在改变我们"),
    ("Smartphones have become an essential part of", "智能手机已经成为"),
    ("The internet was invented in the", "互联网发明于"),
    ("Electric cars are becoming more popular because", "电动汽车越来越受欢迎，因为"),
    ("Social media has changed how people", "社交媒体改变了人们"),
    ("Computers can process information much faster than", "计算机处理信息的速度远快于"),
    ("Robots are being used in factories to", "机器人被用于工厂"),
    ("Virtual reality allows users to", "虚拟现实允许用户"),
    ("The first computer was built in", "第一台计算机建造于"),
    ("Renewable energy sources include solar and", "可再生能源包括太阳能和"),

    # Abstract & General Knowledge
    ("Education is important because it", "教育很重要，因为它"),
    ("The weather forecast says that tomorrow will be", "天气预报说明天将会"),
    ("In many countries, children start school at the age of", "在很多国家，儿童在"),
    ("Doctors recommend that adults drink at least", "医生建议成年人每天至少喝"),
    ("The best way to learn something new is to", "学习新东西的最好方法是"),
    ("Most animals need food, water, and shelter to", "大多数动物需要食物、水和住所来"),
    ("The color of the sky changes because", "天空的颜色会变化，因为"),
    ("Happiness can be found in simple things like", "幸福可以在简单的事物中找到，比如"),
    ("A good leader should be able to", "一个好的领导者应该能够"),
    ("The future of transportation will likely include", "未来的交通工具可能包括"),
]

# Convenience accessors
ENGLISH_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
CHINESE_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
