"""
Name the parts of a mesh with SigLIP2's joint image-text space (the second stage of the
part-naming test; the first stage is diff3f_features.py --save_views --part_labels, which stores
Diff3F's painted views and a part-id mask per view).

SigLIP2 turns an image into one 768-number embedding in two steps: the vision transformer gives a
feature for every 16 x 16 patch, and an attention-pool head (a learned query that attends over the
patch features, followed by a norm and an MLP) combines them into the image embedding, which is the
vector that is compared with text embeddings. Here the head is applied to the patches of one part
only (the patches whose centre pixel lies in that part's mask; a part with fewer than --min_patches
patches in a view is skipped there), which gives a joint-space embedding of the part per view. These
are averaged over the views and L2-normalised, and compared with the text embeddings of "a photo of
the {part} of a {object}" and "a photo of the {part}". The whole object (all foreground patches) is
pooled the same way and compared with "a photo of a {object}" as a check.

Writes <out_dir>/<name>.json (per part: size, views used, the five best part names with the
object's own word and without an object word, with cosine scores; the whole-object check) and
<out_dir>/<name>_parts.npz (part ids and embeddings), and <out_dir>/pairs.json with part-embedding
cosine matrices for --pairs (raw, and after removing each mesh's mean part embedding).

Runs in an environment with open_clip_torch and transformers >= 4.38 (the SigLIP2 tokenizer is a
Gemma tokenizer); see scripts/idarb/requirements-meshfm.txt. Usage (repository root):
    python scripts/idarb/siglip2_part_naming.py --views ~/semtex/partviews \
        --labels results/diff3f/part_labels_tmp --out_dir results/diff3f/part_naming
"""

import argparse
import glob
import json
import os

import numpy as np

OBJECT_WORD = {"labubu_partial": "plush toy", "camel_full": "camel", "cat_partial": "cat",
               "man_full": "man", "superman_partial": "superhero", "lion_full": "lion",
               "cow_partial": "cow", "cow_full": "cow"}
OBJECTS = ["cat", "cow", "lion", "camel", "man", "person", "plush toy", "superhero"]
PARTS = ["head", "face", "eye", "ear", "nose", "mouth", "snout", "neck", "hump", "back", "belly",
         "body", "torso", "chest", "shoulder", "arm", "hand", "front leg", "hind leg", "leg",
         "foot", "paw", "hoof", "tail", "mane", "hood", "hair", "shirt", "trousers", "shoe", "boot",
         "cape", "belt"]
PAIRS = ["labubu_partial:camel_full", "labubu_partial:lion_full", "cat_partial:man_full",
         "superman_partial:camel_full", "cat_partial:lion_full", "cow_partial:cow_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True, help="folder with <name>/view_XX.png and mask_XX.npy")
    ap.add_argument("--labels", required=True,
                    help="folder with <name>_parts.npz (idx, xyz, labels)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--names", nargs="*", default=list(OBJECT_WORD))
    ap.add_argument("--pairs", nargs="*", default=PAIRS)
    ap.add_argument("--min_patches", type=int, default=4)
    args = ap.parse_args()

    import open_clip
    import torch
    from PIL import Image

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2-512", pretrained="webli")
    model = model.eval().cuda()
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-512")
    trunk = model.visual.trunk
    os.makedirs(args.out_dir, exist_ok=True)

    @torch.no_grad()
    def text_emb(prompts):
        out = []
        for i in range(0, len(prompts), 64):
            t = tokenizer(prompts[i:i + 64]).cuda()
            out.append(model.encode_text(t, normalize=True).float().cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def pooled(tokens, sel):
        """The image embedding of the patches sel: the attention-pool head over them only."""
        e = model.visual.head(trunk.forward_head(tokens[:, sel]))
        return torch.nn.functional.normalize(e.float(), dim=-1)[0].cpu().numpy()

    free_prompts = [f"a photo of the {p}" for p in PARTS]
    free_emb = text_emb(free_prompts)
    obj_prompts = [f"a photo of a {o}" for o in OBJECTS]
    obj_emb = text_emb(obj_prompts)
    part_emb = {}
    for name in args.names:
        word = OBJECT_WORD[name]
        own_prompts = [f"a photo of the {p} of a {word}" for p in PARTS]
        own_emb = text_emb(own_prompts)
        labels = np.load(os.path.join(args.labels, f"{name}_parts.npz"))["labels"]
        parts = sorted(int(p) for p in np.unique(labels))
        acc = {p: np.zeros(768) for p in parts + [-1]}
        views_used = {p: 0 for p in parts + [-1]}
        for img_path in sorted(glob.glob(os.path.join(args.views, name, "view_*.png"))):
            mask = np.load(img_path.replace("view_", "mask_").replace(".png", ".npy"))
            img = np.asarray(Image.open(img_path).convert("RGB").resize((512, 512)), np.float32)
            x = torch.as_tensor((img / 255 - 0.5) / 0.5).permute(2, 0, 1)[None].cuda()
            with torch.no_grad():
                tokens = trunk.forward_features(x)  # [1, 1024, 768], 32 x 32 patches
            centre = mask[8::16, 8::16].reshape(-1)  # label at each patch centre, token order
            for p in parts + [-1]:
                sel = np.nonzero(centre >= 0)[0] if p == -1 else np.nonzero(centre == p)[0]
                if len(sel) < args.min_patches:
                    continue
                acc[p] += pooled(tokens, torch.as_tensor(sel).cuda())
                views_used[p] += 1
        emb = {p: acc[p] / max(np.linalg.norm(acc[p]), 1e-8) for p in acc if views_used[p] > 0}

        def top(e, txt, prompts, k=5):
            s = txt @ e
            o = np.argsort(-s)[:k]
            return [[prompts[i], round(float(s[i]), 4)] for i in o]

        out = dict(name=name, object_word=word, n_views=len(glob.glob(
            os.path.join(args.views, name, "view_*.png"))), min_patches=args.min_patches,
            whole_object=dict(views_used=views_used[-1], top_objects=top(emb[-1], obj_emb,
                                                                         obj_prompts, 3)),
            parts=[])
        for p in parts:
            if p not in emb:
                out["parts"].append(dict(part=p, size=int((labels == p).sum()), views_used=0))
                continue
            out["parts"].append(dict(
                part=p, size=int((labels == p).sum()), views_used=views_used[p],
                top_own=top(emb[p], own_emb, own_prompts), top_free=top(emb[p], free_emb,
                                                                       free_prompts),
                embedding=[round(float(v), 6) for v in emb[p]]))
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
            json.dump(out, f, indent=1)
        ids = [p for p in parts if p in emb]
        np.savez(os.path.join(args.out_dir, f"{name}_parts.npz"), part=np.array(ids),
                 emb=np.stack([emb[p] for p in ids]).astype(np.float32),
                 whole=emb[-1].astype(np.float32))
        part_emb[name] = (ids, np.stack([emb[p] for p in ids]))
        print(name, "parts", len(ids), "top-1:",
              [(o["part"], o["top_own"][0][0].split("the ")[1].split(" of")[0])
               for o in out["parts"] if "top_own" in o])

    pairs = {}
    for pair in args.pairs:
        a, b = pair.split(":")
        if a not in part_emb or b not in part_emb:
            continue
        (ia, ea), (ib, eb) = part_emb[a], part_emb[b]
        ca, cb = ea - ea.mean(0), eb - eb.mean(0)
        ca /= np.linalg.norm(ca, axis=1, keepdims=True) + 1e-8
        cb /= np.linalg.norm(cb, axis=1, keepdims=True) + 1e-8
        pairs[pair] = dict(source_parts=ia, target_parts=ib,
                           cos=np.round(ea @ eb.T, 4).tolist(),
                           cos_mean_removed=np.round(ca @ cb.T, 4).tolist())
    with open(os.path.join(args.out_dir, "pairs.json"), "w") as f:
        json.dump(pairs, f, indent=1)
    print("saved", args.out_dir)


if __name__ == "__main__":
    main()
