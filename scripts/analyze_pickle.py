import pickle
import numpy as np
from pathlib import Path
import pprint
from typing import Any, Dict, List
from fire import Fire

def analyze_structure(obj: Any, max_depth: int = 3, current_depth: int = 0) -> str:
    """Analyze the structure of an object recursively."""
    if current_depth >= max_depth:
        return f"{type(obj).__name__}"
    
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return f"Empty {type(obj).__name__}"
        return f"{type(obj).__name__}[{len(obj)}] of {analyze_structure(obj[0], max_depth, current_depth + 1)}"
    
    elif isinstance(obj, dict):
        items = {k: analyze_structure(v, max_depth, current_depth + 1) for k, v in obj.items()}
        return pprint.pformat(items)
    
    elif isinstance(obj, np.ndarray):
        return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
    
    else:
        return f"{type(obj).__name__}"

def print_sample_content(obj: Any) -> None:
    """Print sample content from the object."""
    if isinstance(obj, (list, tuple)):
        print(f"\nFirst item content:")
        pprint.pprint(obj[0])
        print(f"\nNumber of items: {len(obj)}")
    else:
        print("\nContent:")
        pprint.pprint(obj)

def analyze_pickle(pickle_path: str, max_depth: int = 3) -> None:
    """
    Analyze the structure of a pickle file.
    
    Args:
        pickle_path: Path to the pickle file
        max_depth: Maximum depth for structure analysis
    """
    path = Path(pickle_path)
    if not path.exists():
        print(f"File not found: {pickle_path}")
        return
    
    print(f"\nAnalyzing: {path.name}")
    print("-" * 50)
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print("\nStructure:")
    print("-" * 50)
    print(analyze_structure(data, max_depth=max_depth))
    
    print("\nSample Content:")
    print("-" * 50)
    print_sample_content(data)

def main():
    """Analyze both pickle files in the dataset_samples directory."""
    files = [
        "./dataset_samples/brca_hipt_patches.pickle",
        "./dataset_samples/brca_hipt_large_images.pickle"
    ]
    
    for file_path in files:
        analyze_pickle(file_path)

if __name__ == "__main__":
    Fire(main)
