import argparse, torch, clip, faiss, numpy as np
from pathlib import Path
from PIL import Image

def build_index(folder, model, preprocess):
    paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        paths.extend(sorted(Path(folder).glob(ext)))
    embs = []
    for p in paths:
        img = preprocess(Image.open(p)).unsqueeze(0).cuda()
        with torch.no_grad():
            embs.append(model.encode_image(img).cpu().numpy())
    embs = np.concatenate(embs).astype("float32")
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx, [str(p) for p in paths]

def search(idx, paths, query, model, top=10):
    text = clip.tokenize([query]).cuda()
    with torch.no_grad():
        emb = model.encode_text(text).cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)
    scores, ids = idx.search(emb, top)
    return [(paths[i], float(s)) for i, s in zip(ids[0], scores[0])]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args()
    model, preprocess = clip.load("ViT-B/32")
    idx, paths = build_index(a.folder, model, preprocess)
    for path, score in search(idx, paths, a.query, model, a.top):
        print(f"  {score:.3f} | {path}")

if __name__ == "__main__":
    main()
