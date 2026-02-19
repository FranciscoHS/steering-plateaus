#!/usr/bin/env python3
"""
Test which text genres Qwen3-0.6B can reliably auto-complete.

Generates 20-token completions for matched prompts across ~12 candidate genres.
Visual inspection determines which genres the model reliably continues in.

Usage:
    python scripts/registers/test_genre_completions.py [--model qwen3-0.6b]
"""

import sys
import argparse
sys.path.insert(0, '.')

import torch
from src.model import load_model

# --- Candidate genres ---
GENRES = [
    'news',         # news article
    'scientific',   # scientific paper / abstract
    'recipe',       # cooking instructions
    'legal',        # legal / contractual
    'children',     # children's story
    'review',       # product review
    'wikipedia',    # encyclopedia entry
    'dialogue',     # conversation / dialogue
    'email',        # business email
    'diary',        # personal diary / journal
    'technical',    # technical documentation / manual
    'advertising',  # marketing / ad copy
]

# --- Test prompt sets: same rough topic, different genres ---
# Topic 1: Rain / weather
TEST_PROMPTS = [
    {
        'news': "Heavy rainfall caused flooding in several downtown areas on Tuesday, prompting authorities to",
        'scientific': "Precipitation levels exceeding 50mm over a 24-hour period were observed, resulting in significant hydrological",
        'recipe': "If the weather is rainy and cold, this warm soup recipe is perfect. Start by",
        'legal': "In the event of flooding caused by excessive rainfall, the property owner shall be responsible for",
        'children': "Once upon a time, the clouds were so full of rain that they started to",
        'review': "I bought this rain jacket last month and honestly it has been",
        'wikipedia': "Rain is a form of precipitation in which water droplets fall from clouds to the",
        'dialogue': '"Do you think it\'s going to rain today?" she asked, looking at the',
        'email': "Dear Team, Due to the severe weather forecast for tomorrow, we have decided to",
        'diary': "April 12. It rained all day again. I stayed inside and watched the droplets",
        'technical': "The drainage system must be designed to handle a maximum rainfall intensity of 100mm/hr. Ensure that all",
        'advertising': "Don't let the rain stop you! Our new waterproof jacket keeps you dry and comfortable in",
    },
    # Topic 2: Coffee
    {
        'news': "A new study released Monday found that moderate coffee consumption may reduce the risk of",
        'scientific': "Caffeine intake of 200-400mg daily was associated with a statistically significant reduction in",
        'recipe': "For the perfect cold brew coffee, combine one cup of coarsely ground beans with four cups of",
        'legal': "The seller warrants that the coffee beans delivered under this agreement shall conform to the quality standards",
        'children': "Little Bear had never tried coffee before. One morning, he found a big cup and",
        'review': "This coffee maker is absolutely amazing. The brew quality is excellent and it",
        'wikipedia': "Coffee is a brewed drink prepared from roasted coffee beans, the seeds of berries from",
        'dialogue': '"Want some coffee?" he offered, reaching for the pot. "Sure, but make it',
        'email': "Hi Sarah, I wanted to follow up on the coffee supplier contract. Could you please",
        'diary': "March 3. Tried that new coffee shop on the corner today. The espresso was",
        'technical': "The coffee machine should be descaled every 3 months. To begin the descaling procedure, fill the",
        'advertising': "Wake up to perfection. Our premium Colombian roast delivers a rich, smooth flavor that",
    },
    # Topic 3: Dogs
    {
        'news': "A golden retriever named Max was credited with saving a family from a house fire early",
        'scientific': "Canis lupus familiaris exhibits a range of social cognition abilities that facilitate interspecific communication with",
        'recipe': "These homemade dog treats are healthy and easy to make. Combine two cups of oat flour with",
        'legal': "The owner of any dog that causes injury to another person or animal shall be liable for",
        'children': "There was a little puppy named Spot who loved to play in the park. Every day he would",
        'review': "We got this dog bed for our German Shepherd and she absolutely loves it. The",
        'wikipedia': "The domestic dog is a domesticated descendant of the wolf. It was the first species to be",
        'dialogue': '"Your dog is so cute! What breed is he?" "He\'s a mix, mostly',
        'email': "Dear Dr. Williams, I am writing to schedule an appointment for my dog Rex, who has been",
        'diary': "June 7. Took the dog for a long walk by the river today. She was so",
        'technical': "The automatic pet feeder supports programmable feeding schedules for up to 4 meals per day. To configure the",
        'advertising': "Give your best friend the nutrition they deserve. Our grain-free formula is packed with",
    },
    # Topic 4: Space / moon
    {
        'news': "NASA announced Thursday that its latest lunar mission has successfully entered orbit around the moon, marking a",
        'scientific': "Spectroscopic analysis of lunar regolith samples reveals elevated concentrations of titanium dioxide consistent with",
        'recipe': "For these fun moon-shaped cookies, roll out the sugar cookie dough and use a crescent cutter to",
        'legal': "Under the Outer Space Treaty of 1967, no nation may claim sovereignty over the moon or any",
        'children': "The little astronaut put on her helmet and climbed into the rocket ship. She was going to the",
        'review': "This telescope is incredible for the price. I was able to see the craters on the moon",
        'wikipedia': "The Moon is Earth's only natural satellite. It orbits at an average distance of approximately 384,400",
        'dialogue': '"Look at the moon tonight, it\'s beautiful." "Yeah, I read that it\'s a',
        'email': "Dear Professor Chen, I wanted to discuss the preliminary results from the lunar soil analysis. The data",
        'diary': "September 20. Couldn't sleep so I went outside and just stared at the moon. It was",
        'technical': "The lunar rover's navigation system uses a combination of inertial measurement units and star trackers to",
        'advertising': "Reach for the stars with our new telescope! Crystal-clear optics let you explore the moon's",
    },
    # Topic 5: Exercise / running
    {
        'news': "The annual city marathon attracted a record 50,000 participants on Sunday, with the winner finishing in",
        'scientific': "Moderate-intensity aerobic exercise performed three times weekly was shown to significantly improve cardiovascular endurance as measured by",
        'recipe': "After a long run, refuel with this protein smoothie. Blend one banana, a scoop of protein powder, and",
        'legal': "The gym membership agreement shall remain in effect for a period of twelve months from the date of",
        'children': "Rabbit loved to run. She ran through the meadow, over the hill, and all the way to",
        'review': "These running shoes are hands down the best I've ever owned. The cushioning is",
        'wikipedia': "Running is a method of terrestrial locomotion allowing humans and other animals to move rapidly on",
        'dialogue': '"How was your run this morning?" "Tough, but I managed to do five miles in',
        'email': "Hi team, Just a reminder that the company 5K run is next Friday. Please register by",
        'diary': "January 15. Did my first run in weeks today. Only managed 3K but it felt",
        'technical': "The treadmill's incline motor supports gradients from 0% to 15%. To calibrate the incline sensor, press and hold",
        'advertising': "Run faster. Run further. Our new AirStride shoes feature revolutionary carbon-plate technology that",
    },
    # Topic 6: Music
    {
        'news': "The legendary rock band announced their farewell tour on Wednesday, with concerts scheduled across 30 cities",
        'scientific': "Exposure to musical training before age seven was correlated with enhanced white matter connectivity in the",
        'recipe': "Set the mood for dinner with soft background music. While the pasta cooks, prepare the sauce by",
        'legal': "The artist hereby grants the label an exclusive license to reproduce, distribute, and publicly perform the",
        'children': "The little bird sang the most beautiful song in the whole forest. All the animals would stop and",
        'review': "These wireless headphones have amazing sound quality. The bass is deep and the highs are",
        'wikipedia': "Music is an art form whose medium is sound organized in time. Common elements of music include",
        'dialogue': '"What kind of music do you listen to?" "A bit of everything, but mostly',
        'email': "Dear Ms. Johnson, Thank you for your interest in performing at our annual gala. We would like to",
        'diary': "November 8. Listened to that album again tonight. The third track always makes me feel",
        'technical': "The audio interface supports sample rates up to 192kHz at 24-bit resolution. Connect the device via USB-C and",
        'advertising': "Feel every beat. Our premium wireless earbuds deliver studio-quality sound with 30 hours of",
    },
    # Topic 7: Cooking / food
    {
        'news': "A local chef won the prestigious culinary award for her innovative approach to sustainable cooking using",
        'scientific': "The Maillard reaction, occurring between amino acids and reducing sugars at temperatures above 140°C, is responsible for",
        'recipe': "Preheat the oven to 375°F. In a large bowl, mix the flour, sugar, and salt. Cut in the",
        'legal': "The restaurant operator shall comply with all applicable food safety regulations and maintain a valid health",
        'children': "The little chef put on her big white hat and apron. Today she was going to bake a",
        'review': "This cast iron skillet is worth every penny. The heat distribution is incredibly even and",
        'wikipedia': "Cooking is the art and science of preparing food for consumption. It involves the application of heat",
        'dialogue': '"What are you making for dinner?" "I was thinking pasta, maybe with a',
        'email': "Hi everyone, For the office potluck next Thursday, please sign up to bring a dish. So far we",
        'diary': "February 2. Tried making sourdough bread for the first time today. The dough was surprisingly",
        'technical': "The convection oven features a dual-fan system for uniform heat distribution. Set the temperature using the digital",
        'advertising': "Cook like a pro with our premium non-stick cookware set. Restaurant-quality meals made easy in",
    },
    # Topic 8: Sleep
    {
        'news': "Health officials warned Tuesday that chronic sleep deprivation is reaching epidemic levels, with nearly 40 percent of",
        'scientific': "Polysomnographic recordings indicate that REM sleep duration decreases with age, with subjects over 60 exhibiting a mean",
        'recipe': "For a calming bedtime tea, steep one teaspoon of dried chamomile flowers in hot water for",
        'legal': "The employer shall ensure that no employee is required to work more than 12 consecutive hours without a",
        'children': "The moon smiled down at the sleepy little town. One by one, the animals closed their eyes and",
        'review': "This mattress completely changed my sleep quality. After years of back pain, I finally wake up",
        'wikipedia': "Sleep is a naturally recurring state of mind and body characterized by altered consciousness, reduced sensory",
        'dialogue': '"I barely slept last night." "Again? You should really try that melatonin I',
        'email': "Dear HR Department, I am writing to request a schedule adjustment due to a medical sleep condition that",
        'diary': "October 30. Couldn't fall asleep until 3am again. My mind just wouldn't stop racing about",
        'technical': "The sleep tracking sensor uses photoplethysmography to monitor heart rate variability during sleep stages. Place the device",
        'advertising': "Sleep like never before. Our memory foam pillow adapts to your unique shape for the most",
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
    generated = output[0][tokens['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-0.6b')
    parser.add_argument('--n_tokens', type=int, default=25)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer, device = load_model(args.model)
    model.eval()

    print(f"Model: {args.model} on {device}")
    print(f"Generating {args.n_tokens} tokens per prompt, temperature={args.temperature}")
    print(f"Genres: {GENRES}")
    print(f"Test sets: {len(TEST_PROMPTS)}")
    print("=" * 100)

    for i, prompt_set in enumerate(TEST_PROMPTS):
        print(f"\n--- Topic {i+1} ---")
        for genre in GENRES:
            prompt = prompt_set[genre]
            completion = generate_completion(
                model, tokenizer, device, prompt,
                n_tokens=args.n_tokens, temperature=args.temperature,
            )
            # Truncate prompt display to keep output readable
            prompt_display = prompt if len(prompt) <= 70 else prompt[:67] + "..."
            print(f"  [{genre:12s}] {prompt_display} || {completion}")
        print()


if __name__ == '__main__':
    main()
