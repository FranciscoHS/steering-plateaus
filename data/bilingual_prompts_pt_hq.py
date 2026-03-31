"""
28 high-quality hand-crafted English → European Portuguese prompt pairs.
Prompts #5 and #28 from the full set are excluded (intersection with Mistral sets).
"""

PROMPT_PAIRS = [
    # --- Science ---
    ("Because the experiment failed,", "Porque a experiência falhou,"),
    ("The speed of light in a medium", "A velocidade da luz num meio"),
    ("When two atoms collide,", "Quando dois átomos colidem,"),

    # --- Daily life ---
    ("After waking up this morning,", "Depois de acordar esta manhã,"),
    # #5 dropped (intersection)
    ("I usually prefer to", "Normalmente prefiro"),

    # --- History ---
    ("Ancient civilizations often", "As civilizações antigas frequentemente"),
    ("Although the war had ended,", "Embora a guerra tivesse terminado,"),
    ("Long before writing was invented,", "Muito antes de a escrita ser inventada,"),

    # --- Technology ---
    ("The latest software update", "A última atualização de software"),
    ("In order to improve efficiency,", "Para melhorar a eficiência,"),
    ("If the system crashes,", "Se o sistema falhar,"),

    # --- Food ---
    ("If you add too much salt,", "Se puseres sal a mais,"),
    ("Once the water starts boiling,", "Assim que a água começar a ferver,"),
    ("The secret to a good sauce is", "O segredo de um bom molho é"),

    # --- Health ---
    ("Regular exercise can help", "O exercício regular pode ajudar"),
    ("If you don't sleep enough,", "Se não dormires o suficiente,"),
    ("After taking the medicine,", "Depois de tomar o medicamento,"),

    # --- Weather / Nature ---
    ("When the storm finally passes,", "Quando a tempestade finalmente passar,"),
    ("Birds migrate south because", "As aves migram para sul porque"),
    ("Since the river dried up,", "Desde que o rio secou,"),

    # --- Culture / Philosophy ---
    ("The purpose of this tradition is", "O propósito desta tradição é"),
    ("Compared to the old method,", "Comparado com o método antigo,"),
    ("People often forget that", "As pessoas frequentemente esquecem-se de que"),

    # --- Economics ---
    ("If prices keep rising,", "Se os preços continuarem a subir,"),
    ("Ever since the market crashed,", "Desde que o mercado colapsou,"),

    # --- Education / Psychology ---
    ("The best way to learn is", "A melhor forma de aprender é"),
    # #28 dropped (intersection)

    # --- Art / Music ---
    ("The colors in this painting", "As cores nesta pintura"),
    ("Although the melody sounds simple,", "Embora a melodia pareça simples,"),
]

ENGLISH_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
PORTUGUESE_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
