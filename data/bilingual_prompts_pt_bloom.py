"""
bigscience/bloom-1b7 few-shot translations of English prompts to European Portuguese.
Machine-translated for feature purity experiments.
"""

PROMPT_PAIRS = [
    ('Because the experiment failed,', 'Porque o experimento falhou,'),
    ('The speed of light in a medium', 'A velocidade da luz em um meio'),
    ('When two atoms collide,', 'Quando dois átomos se chocam,'),
    ('After waking up this morning,', 'Depois de acordar esta manhã,'),
    ('The easiest way to fix this is', 'A maneira mais fácil de resolver isto é'),
    ('I usually prefer to', 'Eu geralmente prefiro'),
    ('Ancient civilizations often', 'Civilizações antigas costumavam'),
    ('Although the war had ended,', 'Apesar da guerra ter terminado,'),
    ('Long before writing was invented,', 'Antes de escrever ter sido inventado,'),
    ('The latest software update', 'O último update do software.'),
    ('In order to improve efficiency,', 'Para melhorar a eficiência,'),
    ('If the system crashes,', 'Se a máquina falha,'),
    ('If you add too much salt,', 'Se adicionares muito sal,'),
    ('Once the water starts boiling,', 'Quando o líquido começa a ferver,'),
    ('The secret to a good sauce is', 'O segredo de uma boa salada,'),
    ('Regular exercise can help', 'Exercício regular pode ajudar'),
    ("If you don't sleep enough,", 'Se não dormes o suficiente,'),
    ('After taking the medicine,', 'Depois de tomar a medicação,'),
    ('When the storm finally passes,', 'Quando o temporal finalmente passa,'),
    ('Birds migrate south because', 'As aves migram para sul porque'),
    ('Since the river dried up,', 'Desde que o rio secou,'),
    ('The purpose of this tradition is', 'O propósito desta tradição é'),
    ('Compared to the old method,', 'Comparado à antiga maneira,'),
    ('People often forget that', 'Pessoas esquecem que'),
    ('If prices keep rising,', 'Se os preços continuarem a subir,'),
    ('Ever since the market crashed,', 'Desde que o mercado caiu,'),
    ('The best way to learn is', 'A melhor maneira de aprender é'),
    ('Human memory tends to', 'Memória humana tende a'),
    ('The colors in this painting', 'Os tons desta pintura'),
    ('Although the melody sounds simple,', 'Apesar da melodia parecer simples,'),
]

ENGLISH_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
PORTUGUESE_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
