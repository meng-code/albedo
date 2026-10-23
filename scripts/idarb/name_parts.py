"""
Match the parts of two meshes by what they are called, not by how their points look: the first
version of our own method for pairs of different kinds (a toy and a camel).

Each mesh was clustered into parts (part_anchors.py; labels in <name>_parts.npz) and, on the
server, every part was pooled through SigLIP2's attention head over the rendered views, so each
part has one vector in SigLIP2's joint image-text space, next to the text vectors of a part
vocabulary ("a photo of the hump of a camel", ...). This script

  1. reads every part's names: the vocabulary entries closest to its vector (object-free
     prompts, as the server's siglip2_part_naming.py wrote them);
  2. scores every target-source part pair mainly by name agreement (same word, or words of
     the same group such as face and head, over the best three names of both) and partly by
     the cosine of the two part vectors after each object's mean part vector is removed;
  3. keeps the confident one-to-one pairs (bilateral margin, as in part_anchors.py). A target part
     whose name has no counterpart on the source (the hump) stays unmatched, so with
     refine_correspondence.py --parts its vertices follow their neighbours instead of borrowing the
     most similar patch from elsewhere.

Writes the same part_match.npz layout as part_anchors.py (labels_t, labels_s, pairs) and a JSON
with the names and the score matrix.

Usage (repository root, mesh environment):
    python scripts/idarb/name_parts.py \
        --target results/diff3f/part_naming/camel_full.json \
        --target_labels results/diff3f/part_labels_tmp/camel_full_parts.npz \
        --source results/diff3f/part_naming/labubu_partial.json \
        --source_labels results/diff3f/part_labels_tmp/labubu_partial_parts.npz \
        --out <dir>/part_match.npz
"""

import argparse
import json
import os

import numpy as np

from part_anchors import margin_confidence, select_pairs


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


GROUPS = {
    "head": ["head", "face", "eye", "ear", "nose", "mouth", "snout", "hair", "mane", "hood"],
    "leg": ["leg", "front leg", "hind leg", "foot", "paw", "hoof", "shoe", "boot", "trousers"],
    "arm": ["arm", "hand", "shoulder"],
    "body": ["body", "torso", "chest", "belly", "back", "shirt", "cape", "belt"],
    "neck": ["neck"],
    "hump": ["hump"],
    "tail": ["tail"],
}
GROUP_OF = {w: g for g, ws in GROUPS.items() for w in ws}


def word_of(prompt):
    """The part word of a prompt: the text between "the " and " of" (or the end)."""
    w = prompt
    if "the " in w:
        w = w.split("the ", 1)[1]
    if " of " in w:
        w = w.split(" of ", 1)[0]
    return w.strip()


def load_part_file(path):
    """One mesh's parts from the server's JSON (siglip2_part_naming.py): unit part vectors
    (K, D), sizes (K,) and, per part, the object-free names best to worst as (word, score)."""
    with open(path) as f:
        d = json.load(f)
    parts = sorted(d["parts"], key=lambda q: int(q.get("part", q.get("id", 0))))
    emb = unit(np.stack([np.asarray(q["embedding"], dtype=np.float64) for q in parts]))
    size = np.array([float(q.get("size", 1.0)) for q in parts])
    names = [[(word_of(t), float(sc)) for t, sc in q.get("top_free", [])] for q in parts]
    return emb, size, names


def load_vocabulary(path):
    """Prompt strings and their unit text vectors; the part word of each prompt is the text
    between "the " and " of" (or the end), so prompts of both forms give the same word."""
    z = np.load(path, allow_pickle=False)
    text = [str(t) for t in z["text"]]
    emb = unit(z["emb"])
    words = []
    for t in text:
        w = t
        if "the " in w:
            w = w.split("the ", 1)[1]
        if " of " in w:
            w = w.split(" of ", 1)[0]
        words.append(w.strip())
    return text, words, emb


def name_parts(emb, vocab_emb, vocab_words, top=5):
    """For every part, the `top` vocabulary words by cosine (merging duplicates of a word)."""
    S = emb @ vocab_emb.T
    out = []
    for row in S:
        best = {}
        for j in np.argsort(-row):
            w = vocab_words[j]
            if w not in best:
                best[w] = float(row[j])
            if len(best) >= top:
                break
        out.append(list(best.items()))
    return out


def name_similarity(names_a, names_b, weights=(0.5, 0.3, 0.2)):
    """How much two parts' names agree, from their best three names: 1 for the same word, 0.7
    for words of the same group (face and head, leg and hoof), 0 otherwise, averaged with the
    rank weights of both lists. Small margins between a part's best names make the top-1 word
    unreliable, so all three count."""
    total = 0.0
    for wa, (na, _) in zip(weights, names_a[:3]):
        for wb, (nb, _) in zip(weights, names_b[:3]):
            if na == nb:
                total += wa * wb
            elif GROUP_OF.get(na) is not None and GROUP_OF.get(na) == GROUP_OF.get(nb):
                total += 0.7 * wa * wb
    return total


def part_scores(emb_t, emb_s, names_t, names_s, w_name=0.6):
    """Score of every target-source part pair: w_name times the name agreement plus
    (1 - w_name) times the cosine of the two part vectors after each object's mean part vector is
    removed (the plain cosine is nearly the same for every pair, 0.6 to 0.8, so it carries
    nothing)."""
    A = unit(emb_t - emb_t.mean(axis=0))
    B = unit(emb_s - emb_s.mean(axis=0))
    rel = A @ B.T
    N = np.array([[name_similarity(na, nb) for nb in names_s] for na in names_t])
    return w_name * N + (1.0 - w_name) * rel


def match_parts(S, alpha=20, min_conf=0.02, min_score=0.3):
    """Confident one-to-one part pairs: bilateral margin confidence, pairs scoring below
    min_score never qualify (a part with no counterpart must stay unmatched)."""
    conf = margin_confidence(S)
    conf = np.where(S >= min_score, conf, -np.inf)
    return select_pairs(conf, alpha, min_conf)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", required=True, help="part naming JSON of the full mesh")
    ap.add_argument("--target_labels", required=True, help="<name>_parts.npz of the full mesh")
    ap.add_argument("--source", required=True, help="part naming JSON of the relief")
    ap.add_argument("--source_labels", required=True)
    ap.add_argument("--alpha", type=int, default=20, help="at most this many part pairs")
    ap.add_argument("--min_conf", type=float, default=0.02)
    ap.add_argument("--min_score", type=float, default=0.3, help="pairs below never match")
    ap.add_argument("--w_name", type=float, default=0.6, help="weight of the names")
    ap.add_argument("--out", required=True, help="part_match.npz (for refine --parts)")
    args = ap.parse_args()

    emb_t, size_t, names_t = load_part_file(args.target)
    emb_s, size_s, names_s = load_part_file(args.source)
    S = part_scores(emb_t, emb_s, names_t, names_s, args.w_name)
    pairs = match_parts(S, args.alpha, args.min_conf, args.min_score)
    lt = np.load(args.target_labels)["labels"].astype(np.int64)
    ls = np.load(args.source_labels)["labels"].astype(np.int64)
    if lt.max() + 1 != len(emb_t) or ls.max() + 1 != len(emb_s):
        raise ValueError("the part files and the label files have different part counts")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out, labels_t=lt, labels_s=ls,
        pairs=np.array([[a, b, c] for a, b, c in pairs], dtype=np.float64).reshape(-1, 3),
    )
    first = lambda nm: nm[0][0] if nm else "?"  # noqa: E731
    report = dict(
        target_names=[[n for n, _ in nm[:3]] for nm in names_t],
        source_names=[[n for n, _ in nm[:3]] for nm in names_s],
        pairs=[dict(target=a, source=b, confidence=round(c, 4), score=round(float(S[a, b]), 3),
                    target_name=first(names_t[a]), source_name=first(names_s[b]))
               for a, b, c in pairs],
        unmatched_target=[dict(part=j, name=first(names_t[j]), size=float(size_t[j]))
                          for j in range(len(emb_t)) if j not in {a for a, _, _ in pairs}],
        scores=np.round(S, 3).tolist(),
    )
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(dict(pairs=report["pairs"], unmatched_target=report["unmatched_target"])))


if __name__ == "__main__":
    main()
