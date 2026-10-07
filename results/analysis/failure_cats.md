# Failure case: cat2, cat3 and cat4 reconstruct as flat sheets

Figure: `failure_cats.png`. One row per object. From left to right: the input image, the IDArb four-panel
output (albedo, normal, metallic, roughness), the mesh from normals only, and the mesh with the depth
prior. Both meshes are seen from the side (rotated 90 degrees), to show whether the object has any
thickness.

To regenerate the figure (from the repository root, in the `mesh` environment):

```bash
python scripts/idarb/plot_failure_cases.py --names cat2 cat3 cat4 --out results/analysis/failure_cats.png
```

## What we see

Seen from the side, all three cats are a thin line, with or without the depth prior. For cat2 and cat4
the normals-only line is slanted. Their depth ranges (71 and 46 px) come from the whole flat sheet being
tilted, not from thickness.

## Reason 1: IDArb's normals fail on these three images

In the normal maps the three cats are an almost uniform blue-purple. That colour means IDArb predicts that
every point faces the camera. The z component of the normal (the component towards the camera; 1 means
facing the camera head-on) shows this:

| Object | Mean nz | Pixels with nz < 0.9 |
|---|---|---|
| cat2 | 0.983 | 1.1% |
| cat3 | 1.000 | 0.0% |
| cat4 | 0.993 | 0.0% |
| cat (reference) | 0.892 | 35.5% |
| panda (reference) | 0.799 | 62.7% |

The normals say that the surface does not slope anywhere. As a result, the depth integrated from them is
the same everywhere, and the mesh is a flat plate. The same model gives normal results on cat and cat5,
so cats in general do not fail; the failure is specific to these images. We have not yet looked into what
makes IDArb fail on them (fur, viewpoint, lighting or something else).

## Reason 2: when the normals fail, the depth-prior scale s also goes to almost zero

Depth Anything gives only relative depth between 0 and 1. It must be multiplied by a scale s to convert
it to pixels. In `normal_to_mesh.py`, s is not a fixed number. It is fitted by least squares on smooth
regions: s is chosen so that "the depth map's difference between neighbouring pixels, times s" matches
"the difference given by the normals" as closely as possible (step 1 of `integrate_normals_with_depth`).

When the slopes from the normals are close to zero everywhere, the only way to match them is an s close
to zero. This flattens the front-to-back information that the depth map does contain:

| Object | s (px) | Depth range, normals only (px) | Depth range, with depth prior (px) | Object width (px) |
|---|---|---|---|---|
| cat2 | 15.9 | 71.4 | 15.9 | 411 |
| cat3 | 3.0 | 7.0 | 2.5 | 175 |
| cat4 | 3.9 | 46.3 | 4.1 | 289 |
| cat (reference) | 102.0 | 66.8 | 59.1 | 301 |
| panda (reference) | 186.3 | 112.5 | 97.4 | 279 |

On these images the depth prior therefore does not restore any thickness. It pulls the depth towards an
almost flat s·d, which also removes the tilt of the normals-only result, so the result is even flatter
than with normals only.

## Conclusion

This is a limitation of the current method: the depth-prior scale depends on the normals, so when the
normals fail completely, the depth prior fails too. It is recorded here only; `normal_to_mesh.py` was not
changed.

Possible improvements (not done):

- Re-run IDArb with other random seeds or other input preprocessing, to check whether the normal failure
  is consistent.
- If the failure is consistent, stop fitting s to the normals when they have clearly failed, and set the
  scale another way.

Even with a sensible scale, surface detail (fur, the shape of the face) still has to come from the
normals, so the result would only be a smooth outline.

The numbers are printed by `scripts/idarb/plot_failure_cases.py`. It uses the same parameters as the
defaults of `normal_to_mesh.py`: the mask is eroded by 2 px, and the depth-prior weight is 0.01.
