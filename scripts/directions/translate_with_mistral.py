#!/usr/bin/env python3
"""
Use Mistral-7B-Instruct-v0.1 to translate English prompts to European Portuguese.
Saves results to data/bilingual_prompts_pt_mistral.py
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

print("Loading Mistral-7B-Instruct-v0.1...")
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

translations = []
for i, prompt in enumerate(ENGLISH_PROMPTS):
    messages = [
        {"role": "user", "content": f"Translate the following English sentence starter to European Portuguese. Output ONLY the translation, nothing else.\n\nEnglish: {prompt}\nPortuguese:"}
    ]
    # Mistral-Instruct v0.1 uses [INST] ... [/INST] format
    chat_input = f"<s>[INST] {messages[0]['content']} [/INST]"
    inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated tokens
    generated = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    # Take only the first line (in case model generates extra text)
    translation = generated.split('\n')[0].strip()
    print(f"{i+1:2d}. {prompt}")
    print(f"    → {translation}")
    translations.append(translation)

# Save as Python module
output_path = "data/bilingual_prompts_pt_mistral.py"
with open(output_path, 'w') as f:
    f.write('"""\nMistral-7B-Instruct-v0.1 translations of English prompts to European Portuguese.\n')
    f.write('Machine-translated for feature purity experiments.\n"""\n\n')
    f.write('PROMPT_PAIRS = [\n')
    for en, pt in zip(ENGLISH_PROMPTS, translations):
        f.write(f'    ({repr(en)}, {repr(pt)}),\n')
    f.write(']\n\n')
    f.write('ENGLISH_PROMPTS = [pair[0] for pair in PROMPT_PAIRS]\n')
    f.write('PORTUGUESE_PROMPTS = [pair[1] for pair in PROMPT_PAIRS]\n')

print(f"\nSaved {len(translations)} pairs to {output_path}")
