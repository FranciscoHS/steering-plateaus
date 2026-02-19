#!/usr/bin/env python3
"""
Test which language registers Qwen3-0.6B can reliably auto-complete.

Generates 20-token completions for matched prompts across 6 candidate registers:
  formal, casual, academic, simple, literary, slang

Visual inspection determines which registers the model "gets" (continues in the
same register). Registers that work become the basis for the full prompt dataset.

Usage:
    python scripts/registers/test_register_completions.py [--model qwen3-0.6b]
"""

import sys
import argparse
sys.path.insert(0, '.')

import torch
from src.model import load_model

# --- Candidate registers ---
REGISTERS = ['formal', 'casual', 'academic', 'simple', 'literary', 'slang']

# --- Test prompt sets: same semantic content, different registers ---
# Each set covers a different topic to test breadth.

TEST_PROMPTS = [
    # 1. Weather
    {
        'formal': "The meteorological conditions suggest that precipitation is",
        'casual': "Looks like it's gonna rain pretty",
        'academic': "Atmospheric observations indicate imminent precipitation characterized by",
        'simple': "I think it will rain because the sky is",
        'literary': "The heavens gathered their grey curtains, promising a deluge that",
        'slang': "Bro the weather is totally gonna be",
    },
    # 2. Food
    {
        'formal': "The culinary establishment offers an extensive selection of",
        'casual': "This restaurant has a ton of really good",
        'academic': "The gastronomic offerings of this establishment encompass a diverse array of",
        'simple': "The food here is very good and they have",
        'literary': "Upon the table lay a feast of flavors, each dish a",
        'slang': "Dude this place has the most fire",
    },
    # 3. Technology
    {
        'formal': "The latest software update introduces several enhancements to",
        'casual': "So they updated the app and now it can",
        'academic': "The recent software iteration incorporates algorithmic improvements that enhance",
        'simple': "The new update makes the program work better because",
        'literary': "With each iteration, the machine awakened to new possibilities, its",
        'slang': "They dropped a sick update and now this app is",
    },
    # 4. Education
    {
        'formal': "The educational institution has implemented a revised curriculum that",
        'casual': "The school changed up what they teach and now",
        'academic': "Recent pedagogical reforms at the institution have restructured the curriculum to",
        'simple': "The school has new classes that help students learn",
        'literary': "Within those hallowed halls of learning, a new chapter unfolds as",
        'slang': "School totally switched things up and honestly it's kinda",
    },
    # 5. Health
    {
        'formal': "Medical professionals recommend regular physical activity to maintain",
        'casual': "Doctors always say you should exercise more to stay",
        'academic': "Clinical evidence consistently demonstrates that sustained physical activity correlates with",
        'simple': "Moving your body every day helps you stay healthy and",
        'literary': "The body, that temple of breath and bone, thrives when set in",
        'slang': "Docs keep saying hit the gym if you wanna stay",
    },
    # 6. Economy
    {
        'formal': "The quarterly financial indicators suggest a period of moderate",
        'casual': "Money stuff has been kinda weird lately with all the",
        'academic': "Macroeconomic indicators for the fiscal quarter suggest a trajectory of",
        'simple': "The economy is not doing very well right now because",
        'literary': "The markets swayed like ships upon uncertain seas, their fortunes",
        'slang': "The economy is lowkey tanking right now and everyone's",
    },
    # 7. Travel
    {
        'formal': "The tourism sector has experienced substantial growth in international",
        'casual': "More people are traveling these days and it's getting",
        'academic': "Quantitative analysis of tourism data reveals a significant upward trend in",
        'simple': "Many people like to travel to new places because it is",
        'literary': "Beyond the horizon, the wanderer's heart beckons toward lands where",
        'slang': "Travel is so lit right now everyone's out there",
    },
    # 8. Environment
    {
        'formal': "Environmental regulations have been strengthened to address the growing",
        'casual': "They're making stricter rules about pollution because things are getting",
        'academic': "Regulatory frameworks governing environmental protection have undergone significant revision to",
        'simple': "We need to take care of the earth because pollution is",
        'literary': "The earth, patient mother, bears the weight of our excess as",
        'slang': "People are finally getting serious about saving the planet because it's",
    },
    # 9. Social media
    {
        'formal': "Social media platforms have fundamentally altered the manner in which",
        'casual': "Social media has totally changed how people talk to each",
        'academic': "Digital communication platforms have precipitated a paradigmatic shift in interpersonal",
        'simple': "People use the internet to talk to friends and share",
        'literary': "In the luminous web of connection, each voice becomes a",
        'slang': "Social media is wild these days like everyone's just",
    },
    # 10. Work
    {
        'formal': "The organization has adopted flexible working arrangements to accommodate",
        'casual': "Work from home is way more common now and honestly",
        'academic': "Organizational restructuring toward hybrid work modalities reflects an adaptation to",
        'simple': "Many people now work from their houses and they like it",
        'literary': "The office, once a fortress of routine, now yields to the",
        'slang': "WFH is the move fr, nobody wants to go back to",
    },
]


def generate_completion(model, tokenizer, device, prompt, n_tokens=20, temperature=0.7):
    """Generate a completion for a given prompt."""
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            **tokens,
            max_new_tokens=n_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated tokens
    generated = output[0][tokens['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--n_tokens', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer, device = load_model(args.model)
    model.eval()

    print(f"Model: {args.model} on {device}")
    print(f"Generating {args.n_tokens} tokens per prompt, temperature={args.temperature}")
    print(f"Registers: {REGISTERS}")
    print(f"Test sets: {len(TEST_PROMPTS)}")
    print("=" * 80)

    for i, prompt_set in enumerate(TEST_PROMPTS):
        print(f"\n--- Prompt set {i+1} ---")
        for reg in REGISTERS:
            prompt = prompt_set[reg]
            completion = generate_completion(
                model, tokenizer, device, prompt,
                n_tokens=args.n_tokens, temperature=args.temperature,
            )
            print(f"  [{reg:10s}] {prompt} || {completion}")
        print()


if __name__ == '__main__':
    main()
