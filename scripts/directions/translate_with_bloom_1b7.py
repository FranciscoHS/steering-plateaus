#!/usr/bin/env python3
"""
Use bloom-1b7 to translate English prompts to European Portuguese via few-shot prompting.
Saves results to data/bilingual_prompts_pt_bloom.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ENGLISH_PROMPTS = [
    # Science
    "Because the experiment failed,",
    "The speed of light in a medium",
    "When two atoms collide,",
    # Daily life
    "After waking up this morning,",
    "The easiest way to fix this is",
    "I usually prefer to",
    # History
    "Ancient civilizations often",
    "Although the war had ended,",
    "Long before writing was invented,",
    # Technology
    "The latest software update",
    "In order to improve efficiency,",
    "If the system crashes,",
    # Food
    "If you add too much salt,",
    "Once the water starts boiling,",
    "The secret to a good sauce is",
    # Health
    "Regular exercise can help",
    "If you don't sleep enough,",
    "After taking the medicine,",
    # Weather / Nature
    "When the storm finally passes,",
    "Birds migrate south because",
    "Since the river dried up,",
    # Culture / Philosophy
    "The purpose of this tradition is",
    "Compared to the old method,",
    "People often forget that",
    # Economics
    "If prices keep rising,",
    "Ever since the market crashed,",
    # Education / Psychology
    "The best way to learn is",
    "Human memory tends to",
    # Art / Music
    "The colors in this painting",
    "Although the melody sounds simple,",
]

# Few-shot examples to guide the model
FEW_SHOT = """Translate English to Portuguese.

English: The cat is sleeping on the table.
Portuguese: O gato está a dormir em cima da mesa.

English: If it rains tomorrow,
Portuguese: Se chover amanhã,

English: She opened the door slowly.
Portuguese: Ela abriu a porta devagar.

English: The children played in the garden until sunset.
Portuguese: As crianças brincaram no jardim até ao pôr do sol.

English: Before starting the project,
Portuguese: Antes de começar o projeto,

"""

print("Loading bloom-1b7...")
model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

translations = []
for i, prompt in enumerate(ENGLISH_PROMPTS):
    input_text = FEW_SHOT + f"English: {prompt}\nPortuguese:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # Take the first line, strip whitespace
    translation = generated.split('\n')[0].strip()
    # If model generated "English:" on the same line, truncate before it
    if 'English:' in translation:
        translation = translation[:translation.index('English:')].strip()
    print(f"{i+1:2d}. {prompt}")
    print(f"    → {translation}")
    translations.append(translation)

# Save as Python module
output_path = "data/bilingual_prompts_pt_bloom.py"
with open(output_path, 'w') as f:
    f.write('"""\nbigscience/bloom-1b7 few-shot translations of English prompts to European Portuguese.\n')
    f.write('Machine-translated for feature purity experiments.\n"""\n\n')
    f.write('PROMPT_PAIRS = [\n')
    for en, pt in zip(ENGLISH_PROMPTS, translations):
        f.write(f'    ({repr(en)}, {repr(pt)}),\n')
    f.write(']\n\n')
    f.write('ENGLISH_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]\n')
    f.write('PORTUGUESE_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]\n')

print(f"\nSaved {len(translations)} pairs to {output_path}")
