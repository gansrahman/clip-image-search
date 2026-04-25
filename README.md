# CLIP Image Search

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenAI CLIP](https://img.shields.io/badge/OpenAI-CLIP-green.svg)](https://github.com/openai/CLIP)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Semantic image search engine powered by **OpenAI CLIP** embeddings. Search your local image library with natural language queries.

## Features

- **Semantic search** - find images by meaning, not just filenames
- **FAISS-powered** fast similarity search (millions of images)
- Support for JPEG, PNG, WebP formats
- Batch indexing with progress tracking
- Adjustable top-k results

## Installation

```bash
git clone https://github.com/gansrahman/clip-image-search.git
cd clip-image-search
pip install -r requirements.txt
```

## Quick Start

```bash
# Index and search
python search.py --folder ./my_photos/ --query "sunset over mountains" --top 10

# Search with different CLIP model
python search.py --folder ./photos/ --query "a dog playing" --model ViT-L/14
```

## Python API

```python
import clip
from search import build_index, search

model, preprocess = clip.load("ViT-B/32")
index, paths = build_index("./photos/", model, preprocess)
results = search(index, paths, "beach sunset", model, top=5)

for path, score in results:
    print(f"{score:.3f} | {path}")
```

## How It Works

1. **Indexing**: Each image is encoded into a 512-dim vector using CLIP's image encoder
2. **Search**: Text query is encoded using CLIP's text encoder
3. **Retrieval**: FAISS finds nearest neighbors via cosine similarity

## Performance

| Dataset Size | Index Time | Search Time |
|-------------|-----------|-------------|
| 1,000 images | ~30s | <10ms |
| 10,000 images | ~5min | <10ms |
| 100,000 images | ~50min | <50ms |

## License

MIT
