import argparse
from collections import Counter

from utils.dataset import get_pathvqa, get_slake, get_omni_rad_closed


DATASET_LOADERS = {
    "pathvqa": get_pathvqa,
    "slake": get_slake,
    "omni-rad-closed": get_omni_rad_closed,
}


def main():
    parser = argparse.ArgumentParser(description="Inspect a processed medical VQA dataset split.")
    parser.add_argument("--dataset", choices=DATASET_LOADERS, default="slake")
    parser.add_argument("--split", default="test")
    parser.add_argument("--truncate", type=int, default=5)
    parser.add_argument("--answer_type", default="all")
    parser.add_argument("--content_type", default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loader = DATASET_LOADERS[args.dataset]
    data = loader(
        truncate=args.truncate,
        split=args.split,
        answer_type=args.answer_type,
        content_type=args.content_type,
        seed=args.seed,
    )

    print(f"dataset: {args.dataset}")
    print(f"split: {args.split}")
    print(f"samples: {len(data)}")

    answer_types = Counter(item.get("answer_type", "unknown") for item in data)
    content_types = Counter(item.get("content_type", "unknown") for item in data)
    print(f"answer_type distribution: {dict(answer_types)}")
    print(f"content_type distribution: {dict(content_types)}")

    if data:
        sample = dict(data[0])
        sample.pop("prompt", None)
        print("first sample metadata:")
        for key, value in sample.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
