"""
50 semantically matched English-Spanish prompt pairs for language steering experiments.

Each pair expresses the same meaning, designed to elicit natural continuations
in the respective language. Matches the English prompts from bilingual_prompts.py.
"""

PROMPT_PAIRS = [
    # Science & Nature
    ("The sun rises in the", "El sol sale por el"),
    ("Water boils at a temperature of", "El agua hierve a una temperatura de"),
    ("The largest planet in our solar system is", "El planeta más grande de nuestro sistema solar es"),
    ("Trees produce oxygen through", "Los árboles producen oxígeno a través de"),
    ("The speed of light is approximately", "La velocidad de la luz es aproximadamente"),
    ("Earthquakes are caused by", "Los terremotos son causados por"),
    ("The human brain contains about", "El cerebro humano contiene aproximadamente"),
    ("Diamonds are made of", "Los diamantes están hechos de"),
    ("The deepest ocean in the world is", "El océano más profundo del mundo es"),
    ("Photosynthesis is the process by which", "La fotosíntesis es el proceso por el cual"),

    # Geography & History
    ("The capital of Japan is", "La capital de Japón es"),
    ("The Great Wall of China was built to", "La Gran Muralla China fue construida para"),
    ("The longest river in Africa is", "El río más largo de África es"),
    ("The population of India is approximately", "La población de India es aproximadamente"),
    ("Mount Everest is located in", "El Monte Everest se encuentra en"),
    ("The ancient Egyptians built pyramids to", "Los antiguos egipcios construyeron pirámides para"),
    ("The Pacific Ocean is the", "El Océano Pacífico es el"),
    ("The first person to walk on the moon was", "La primera persona en caminar sobre la luna fue"),
    ("Australia is known for its", "Australia es conocida por sus"),
    ("The Sahara Desert covers an area of", "El desierto del Sahara cubre un área de"),

    # Daily Life
    ("Breakfast is the most important meal of", "El desayuno es la comida más importante del"),
    ("Many people enjoy drinking coffee in the", "Muchas personas disfrutan tomando café por la"),
    ("Exercise is important because it helps", "El ejercicio es importante porque ayuda a"),
    ("Reading books can improve your", "Leer libros puede mejorar tu"),
    ("A balanced diet should include", "Una dieta equilibrada debe incluir"),
    ("Getting enough sleep is essential for", "Dormir lo suficiente es esencial para"),
    ("Learning a new language requires", "Aprender un nuevo idioma requiere"),
    ("Cooking at home is healthier than", "Cocinar en casa es más saludable que"),
    ("Music can help people feel more", "La música puede ayudar a las personas a sentirse más"),
    ("Traveling to new places helps us", "Viajar a nuevos lugares nos ayuda a"),

    # Technology
    ("Artificial intelligence is changing the way we", "La inteligencia artificial está cambiando la forma en que"),
    ("Smartphones have become an essential part of", "Los teléfonos inteligentes se han convertido en una parte esencial de"),
    ("The internet was invented in the", "Internet fue inventado en los"),
    ("Electric cars are becoming more popular because", "Los coches eléctricos son cada vez más populares porque"),
    ("Social media has changed how people", "Las redes sociales han cambiado la forma en que las personas"),
    ("Computers can process information much faster than", "Las computadoras pueden procesar información mucho más rápido que"),
    ("Robots are being used in factories to", "Los robots se están utilizando en fábricas para"),
    ("Virtual reality allows users to", "La realidad virtual permite a los usuarios"),
    ("The first computer was built in", "La primera computadora fue construida en"),
    ("Renewable energy sources include solar and", "Las fuentes de energía renovable incluyen la solar y"),

    # Abstract & General Knowledge
    ("Education is important because it", "La educación es importante porque"),
    ("The weather forecast says that tomorrow will be", "El pronóstico del tiempo dice que mañana será"),
    ("In many countries, children start school at the age of", "En muchos países, los niños comienzan la escuela a la edad de"),
    ("Doctors recommend that adults drink at least", "Los médicos recomiendan que los adultos beban al menos"),
    ("The best way to learn something new is to", "La mejor manera de aprender algo nuevo es"),
    ("Most animals need food, water, and shelter to", "La mayoría de los animales necesitan comida, agua y refugio para"),
    ("The color of the sky changes because", "El color del cielo cambia porque"),
    ("Happiness can be found in simple things like", "La felicidad se puede encontrar en cosas simples como"),
    ("A good leader should be able to", "Un buen líder debe ser capaz de"),
    ("The future of transportation will likely include", "El futuro del transporte probablemente incluirá"),
]

# Convenience accessors
ENGLISH_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]
SPANISH_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]
