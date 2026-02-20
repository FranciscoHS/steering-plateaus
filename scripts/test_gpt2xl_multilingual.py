#!/usr/bin/env python3
"""Quick test: can GPT-2 XL complete text in multiple languages?"""

import sys
sys.path.insert(0, '.')

import torch
from src.model import load_model

prompts = {
    'English': [
        "The weather today is beautiful, so I decided to",
        "In recent years, technology has changed the way we",
        "The president announced that the new policy will",
    ],
    'Spanish': [
        "El clima de hoy es hermoso, así que decidí",
        "En los últimos años, la tecnología ha cambiado la forma en que",
        "El presidente anunció que la nueva política",
    ],
    'Mandarin': [
        "今天的天气很好，所以我决定",
        "近年来，科技改变了我们",
        "总统宣布新政策将会",
    ],
    'Arabic': [
        "الطقس اليوم جميل، لذلك قررت أن",
        "في السنوات الأخيرة، غيرت التكنولوجيا طريقة",
        "أعلن الرئيس أن السياسة الجديدة ستكون",
    ],
}

def main():
    print("Loading GPT-2 XL...")
    model, tokenizer, device = load_model('gpt2-xl')

    for lang, lang_prompts in prompts.items():
        print(f"\n{'='*60}")
        print(f"  {lang}")
        print(f"{'='*60}")
        for prompt in lang_prompts:
            tokens = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(
                    **tokens,
                    max_new_tokens=20,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(out[0], skip_special_tokens=True)
            # Show prompt vs completion
            new_text = completion[len(tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)):]
            print(f"\n  Prompt: {prompt}")
            print(f"  Completion: {new_text}")

    print("\nDone!")

if __name__ == '__main__':
    main()
