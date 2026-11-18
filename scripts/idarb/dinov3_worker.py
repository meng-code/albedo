"""
DINOv3 patch features as a separate process, for diff3f_features.py --dino dinov3.

DINOv3 needs transformers >= 4.56, which needs torch >= 2.2, while Diff3F's environment is pinned
to torch 2.1 (PyTorch3D is built against it). So diff3f_features.py starts this script with the
Python of an environment that has both (the trellis2 environment on the GPU server: torch 2.6,
transformers 4.57.1) and talks to it through pipes: it writes each painted view as a uint8
H x W x 3 array to this process's stdin, and reads back the last layer's normalised patch tokens
as a float16 n x n x 768 array (the CLS and register tokens dropped). An array goes as one text
line "<dtype> <shape...>" followed by its raw bytes (numpy's .npy reader needs a seekable file).

The image is resized to --size px (592 gives the same 37 x 37 patch grid as Diff3F's DINOv2 at
518 px with 14-px patches) and normalised with the ImageNet mean and standard deviation, as
Diff3F's get_dino_features does for DINOv2. Nothing but arrays is written to stdout.

Not run by hand. The model is gated on Hugging Face (DINOv3 License) and needs a login with
access granted.
"""

import argparse
import sys

import numpy as np


def send_array(fp, a):
    a = np.ascontiguousarray(a)
    fp.write((" ".join([a.dtype.str] + [str(d) for d in a.shape]) + "\n").encode())
    fp.write(a.tobytes())
    fp.flush()


def recv_array(fp):
    """The next array from a pipe, or None at end of input."""
    line = fp.readline()
    if not line:
        return None
    dtype, *shape = line.decode().split()
    shape = [int(d) for d in shape]
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = bytearray()
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            raise EOFError("pipe closed inside an array")
        buf += chunk
    return np.frombuffer(bytes(buf), dtype=dtype).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--size", type=int, default=592)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms as tfs
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model).eval().cuda()
    n = args.size // model.config.patch_size
    transform = tfs.Compose([tfs.Resize((args.size, args.size)), tfs.ToTensor(),
                             tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        img = recv_array(inp)
        if img is None:
            break
        x = transform(Image.fromarray(img))[:3].unsqueeze(0).cuda()
        with torch.no_grad():
            tokens = model(pixel_values=x).last_hidden_state[0, -n * n:]
        send_array(out, tokens.reshape(n, n, -1).half().cpu().numpy())


if __name__ == "__main__":
    main()
