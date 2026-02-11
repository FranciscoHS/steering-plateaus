"""
30 semantically matched English-Dutch prompt pairs for language steering experiments.

Each pair expresses the same meaning (~5 tokens each), designed to elicit natural
continuations in the respective language. Dutch and English are closely related
(both West Germanic), so this tests whether steering works for nearby languages
too, not just distant ones like English-Chinese.
"""

PROMPT_PAIRS = [
    # Science & Nature
    ("The sun is warm", "De zon is warm"),
    ("Water is very important", "Water is erg belangrijk"),
    ("The weather today is", "Het weer vandaag is"),
    ("The tree is big", "De boom is groot"),
    ("The sea is very", "De zee is heel"),

    # Daily Life
    ("I like to read", "Ik lees heel graag"),
    ("She goes to the", "Zij gaat naar de"),
    ("We live in a", "Wij wonen in een"),
    ("He works at the", "Hij werkt bij de"),
    ("The children play outside", "De kinderen spelen buiten"),

    # Food & Drink
    ("I always drink coffee", "Ik drink altijd koffie"),
    ("The food here is", "Het eten hier is"),
    ("We eat together at", "Wij eten samen om"),
    ("She is very old", "Zij is heel oud"),
    ("The bread is fresh", "Het brood is vers"),

    # Places & Travel
    ("The city has many", "De stad heeft veel"),
    ("The train is late", "De trein is laat"),
    ("My house is very", "Mijn huis is heel"),
    ("I come from a", "Ik kom uit een"),
    ("They walk through the", "Zij lopen door het"),

    # Descriptions
    ("The old house has", "Het oude huis heeft"),
    ("The flowers are very", "De bloemen zijn heel"),
    ("The song is nice", "Het lied is mooi"),
    ("The winter is always", "De winter is altijd"),
    ("The book is about", "Het boek gaat over"),

    # Actions
    ("I have a small", "Ik heb een klein"),
    ("The dog is sweet", "De hond is lief"),
    ("The cat sleeps on", "De kat slaapt op"),
    ("The bird is white", "De vogel is wit"),
    ("We need much more", "Wij hebben veel meer"),
]

# Convenience accessors (match interface of bilingual_prompts.py)
ENGLISH_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
DUTCH_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
