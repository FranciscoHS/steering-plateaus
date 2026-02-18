#!/usr/bin/env python3
"""Check token lengths for animal breed/species prompts across categories."""

from transformers import AutoTokenizer

# ---- Candidate prompts ----
DOGS = [
    "Golden Retriever",
    "German Shepherd",
    "Border Collie",
    "Siberian Husky",
    "Cocker Spaniel",
    "English Bulldog",
    "Shih Tzu",
    "Great Dane",
    "Basset Hound",
    "Irish Setter",
    "Afghan Hound",
    "Welsh Corgi",
    "Scottish Terrier",
    "Bernese Mountain",
    "English Mastiff",
    "French Bulldog",
    "Italian Greyhound",
    "Australian Shepherd",
    "English Setter",
    "Alaskan Malamute",
]

CATS = [
    "British Shorthair",
    "Maine Coon",
    "Persian Cat",
    "Bengal Cat",
    "Siamese Cat",
    "Ragdoll Cat",
    "Russian Blue",
    "Scottish Fold",
    "Norwegian Forest",
    "Abyssinian Cat",
    "Burmese Cat",
    "Turkish Angora",
    "Egyptian Mau",
    "Cornish Rex",
    "Devon Rex",
    "Havana Brown",
    "Somali Cat",
    "Tonkinese Cat",
    "Chartreux Cat",
    "Birman Cat",
]

BIRDS = [
    "Bald Eagle",
    "Peregrine Falcon",
    "Great Horned Owl",
    "Ruby Hummingbird",
    "Atlantic Puffin",
    "Barn Owl",
    "Blue Jay",
    "Cedar Waxwing",
    "Golden Eagle",
    "House Sparrow",
    "Indigo Bunting",
    "Scarlet Macaw",
    "Snowy Owl",
    "Wandering Albatross",
    "Wood Thrush",
    "Barn Swallow",
    "Eastern Bluebird",
    "Great Blue Heron",
    "Northern Cardinal",
    "Common Raven",
]

FISH = [
    "Porcupine Puffer",
    "Great White Shark",
    "Bluefin Tuna",
    "Clown Fish",
    "Neon Tetra",
    "Tiger Barb",
    "Manta Ray",
    "Whale Shark",
    "Rainbow Trout",
    "Spotted Grouper",
    "Butterfly Fish",
    "Hammerhead Shark",
    "Electric Eel",
    "Emperor Angelfish",
    "Blue Marlin",
    "Swordfish",
    "Ocean Sunfish",
    "Sockeye Salmon",
    "Mandarin Goby",
    "Largemouth Bass",
]

SNAKES = [
    "King Cobra",
    "Black Mamba",
    "Green Anaconda",
    "Coral Snake",
    "Rough Green Snake",
    "Burmese Python",
    "Boa Constrictor",
    "Garter Snake",
    "Eastern Diamondback",
    "Copperhead Snake",
    "Corn Snake",
    "Ball Python",
    "Milk Snake",
    "Bull Snake",
    "Cottonmouth Snake",
    "Indigo Snake",
    "Rat Snake",
    "Water Moccasin",
    "Sidewinder Snake",
    "Reticulated Python",
]

ALL_CATEGORIES = {
    "Dogs": DOGS,
    "Cats": CATS,
    "Birds": BIRDS,
    "Fish": FISH,
    "Snakes": SNAKES,
}

def main():
    print("Loading Qwen3-0.6B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    for cat_name, prompts in ALL_CATEGORIES.items():
        lengths = []
        print(f"\n{cat_name}:")
        for p in prompts:
            toks = tokenizer.encode(p)
            lengths.append(len(toks))
            print(f"  {len(toks)} tokens: {p}  ->  {tokenizer.convert_ids_to_tokens(toks)}")
        print(f"  Stats: min={min(lengths)}, max={max(lengths)}, "
              f"mean={sum(lengths)/len(lengths):.1f}, median={sorted(lengths)[len(lengths)//2]}")

    # Overall comparison
    print(f"\n{'='*60}")
    print("Category comparison:")
    for cat_name, prompts in ALL_CATEGORIES.items():
        lengths = [len(tokenizer.encode(p)) for p in prompts]
        print(f"  {cat_name}: mean={sum(lengths)/len(lengths):.1f}, "
              f"min={min(lengths)}, max={max(lengths)}")


if __name__ == '__main__':
    main()
