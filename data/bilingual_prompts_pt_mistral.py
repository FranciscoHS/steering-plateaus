"""
Mistral-7B-Instruct-v0.1 translations of English prompts to European Portuguese.
Machine-translated for feature purity experiments.
"""

PROMPT_PAIRS = [
    ('Because the experiment failed,', 'Por falhar da experiência.'),
    ('The speed of light in a medium', 'A velocidade da luz em um meio'),
    ('When two atoms collide,', '"Quando dois átomos colidem,"'),
    ('After waking up this morning,', 'Depois de acordar esta manhã,'),
    ('The easiest way to fix this is', 'The easiest way to fix this is'),
    ('I usually prefer to', 'Gosto geralmente de'),
    ('Ancient civilizations often', 'Civilizações antigas frequentemente'),
    ('Although the war had ended,', '"Embora a guerra tivesse terminado,"'),
    ('Long before writing was invented,', 'Muito antes de ser inventado o escritório.'),
    ('The latest software update', 'A última atualização de software'),
    ('In order to improve efficiency,', 'Para melhorar a eficácia,'),
    ('If the system crashes,', 'Se o sistema caí,'),
    ('If you add too much salt,', 'Se adicionas demasiado sal,'),
    ('Once the water starts boiling,', '"Quando o água começa a ferver,"'),
    ('The secret to a good sauce is', 'O segredo da boa molho é'),
    ('Regular exercise can help', '"Regular exercise can help" translates to "A exercício regular pode ajudar" in European Portuguese.'),
    ("If you don't sleep enough,", 'Se não dormir bem,'),
    ('After taking the medicine,', 'Depois de tomar a medicina,'),
    ('When the storm finally passes,', '"Quando a tempestade finalmente passa,"'),
    ('Birds migrate south because', 'Portuguese: As aves migram para o sul por causa de'),
    ('Since the river dried up,', '"Desde que o rio secou,"'),
    ('The purpose of this tradition is', 'O propósito desta tradição é'),
    ('Compared to the old method,', 'Comparado ao método antigo,'),
    ('People often forget that', '"Pessoas frequentemente esquecem que"'),
    ('If prices keep rising,', 'Se os preços continuarem a subir,'),
    ('Ever since the market crashed,', '"Desde que o mercado caiu."'),
    ('The best way to learn is', 'O melhor modo de aprender é'),
    ('Human memory tends to', 'Human memory tends to'),
    ('The colors in this painting', 'As cores deste quadro'),
    ('Although the melody sounds simple,', '"Embora a melodia somne simples, "'),
]

ENGLISH_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
PORTUGUESE_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
