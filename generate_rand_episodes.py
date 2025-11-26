import pandas as pd
import numpy as np

def generate_episode_csv(
        path="episodes.csv",
        count=10000,
        min_size=32,
        max_size=128,
        seed=None):

    rng = np.random.default_rng(seed)

    widths  = rng.integers(min_size, max_size + 1, size=count)
    heights = rng.integers(min_size, max_size + 1, size=count)
    seeds   = rng.integers(0, 10_000_000, size=count)

    df = pd.DataFrame({
        "width": widths,
        "height": heights,
        "env_seed": seeds
    })

    df.to_csv(path, index=False)
    print(f"✔ Generated {count} episodes → {path}")


if __name__ == "__main__":
    generate_episode_csv(
        path="episodes.csv",
        count=5000,
        min_size=32,
        max_size=128,
        seed=None
    )
