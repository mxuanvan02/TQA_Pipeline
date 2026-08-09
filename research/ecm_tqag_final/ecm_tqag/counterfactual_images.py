#!/usr/bin/env python3
"""Structure-preserving counterfactual page images (ECM-TQAG, W3).

WHY IMAGE DELETION IS THE WRONG INTERVENTION
--------------------------------------------
The v1 ablation (`visual_dependence.py`) deletes the page image and re-asks. That
intervention removes THREE things at once: (a) 2-D layout, (b) figure/table
structure, (c) every glyph that exists only as pixels. If the item's declared
visual observation happens to be a measurable function of the OCR text T -- i.e.
there is a deterministic g with observation = g(T) -- then the answerer can
recover the observation from T alone and

    VDR = 0

is a LOGICAL CONSEQUENCE of the item, not evidence about vision. On
runs/full_v6_dependency the accuracy was 1.0 in both arms, so the measurement had
no sensitivity at all.

THE REPLACEMENT
---------------
Let pi: I -> T be the OCR projection. We construct a counterfactual image I' with

    pi(I') ~= pi(I)   as a MULTISET of tokens,     but
    layout(I') != layout(I)

so the token inventory is held (approximately) fixed and only the 2-D arrangement
is perturbed. Every perturbation here is implemented as a PERMUTATION OF PIXELS,
which gives the multiset guarantee at the strongest available level: the pixel
histogram of I' is bit-exactly equal to that of I (see `histogram_l1`). Since
each glyph is moved intact, the glyph inventory -- hence the token multiset up to
reading order -- is preserved.

    NDE_I = E[acc(a | T, I)] - E[acc(a | T, I')]

is then a natural direct effect of image STRUCTURE with the token content held
fixed. In the degenerate regime observation = g(T) the answerer needs no spatial
binding, so NDE_I = 0 identically. Any item with NDE_I > 0 must have used
position-to-content binding that T does not carry. This is strictly stronger than
VDR: VDR > 0 implies the image mattered for SOME reason (possibly only because it
carried glyphs missing from T), whereas NDE_I > 0 isolates structure.

WHAT THIS DOES NOT CLAIM
------------------------
We do not run OCR on I'. The multiset claim is a claim about pixels and glyphs,
not a claim about what a particular OCR engine emits: a real engine reading I'
will very likely emit the same tokens in a DIFFERENT reading order, and for P2 may
merge or split tokens at block seams. That residual is recorded, not hidden
(`seam_count`, `notes`), and it biases NDE_I UPWARD, so a null result -- the
expected outcome for degenerate items -- is not explained away by it.

PERTURBATION FAMILIES
---------------------
P1 LABEL_PERMUTE : detect text lines, then swap the pixel content of two
                   equal-sized, disjoint line regions. Token multiset exactly
                   preserved; the position-to-label binding is destroyed
                   (the label that was at the top is now further down).
P2 BLOCK_SHUFFLE : partition the largest centred k*k-divisible region into k*k
                   equal blocks and permute them with the documented LCG below.
                   Token multiset exactly preserved; global geometry destroyed.
P3 REGION_OCCLUDE: black out exactly the declared bbox. This is NOT a structure
                   permutation -- it deletes content and does NOT preserve the
                   histogram -- so it is a targeted NECESSITY probe and is
                   labelled separately everywhere (`family="P3_REGION_OCCLUDE"`,
                   `preserves_multiset=False`). It must never be pooled with
                   P1/P2 into an NDE.

DETERMINISM
-----------
Randomness comes from `Lcg`, a self-contained 32-bit linear congruential
generator (Numerical Recipes constants), NOT from `random`. Reason: the expected
block order in the test suite must be derivable BY HAND from the published
recurrence, and must not change if CPython's Mersenne Twister internals change.

    x_{n+1} = (1664525 * x_n + 1013904223) mod 2**32

Shuffling is Durstenfeld (Fisher-Yates from the end), one draw per step.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment probe
    raise ImportError(
        "BLOCKED_IMAGING:pillow_unavailable:install Pillow into a venv "
        "(PEP 668 forbids system pip on this host)"
    ) from exc

SCHEMA = "ecm-tqag.counterfactual-image.v1"

FAMILY_P1 = "P1_LABEL_PERMUTE"
FAMILY_P2 = "P2_BLOCK_SHUFFLE"
FAMILY_P3 = "P3_REGION_OCCLUDE"

# P3 is deliberately excluded: it deletes content, so it is not a
# structure-preserving counterfactual and cannot enter an NDE_I.
NDE_FAMILIES = (FAMILY_P1, FAMILY_P2)

LCG_A = 1664525
LCG_C = 1013904223
LCG_M = 2 ** 32


class Lcg:
    """32-bit LCG: x <- (1664525 x + 1013904223) mod 2**32. Spec'd, hand-checkable."""

    __slots__ = ("state", "_draws")

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("BLOCKED_SEED:seed_must_be_int")
        if seed < 0:
            raise ValueError("BLOCKED_SEED:seed_must_be_nonnegative")
        self.state = seed % LCG_M
        self._draws = 0

    def next_u32(self) -> int:
        self.state = (LCG_A * self.state + LCG_C) % LCG_M
        self._draws += 1
        return self.state

    def below(self, bound: int) -> int:
        """Uniform-ish index in [0, bound). Modulo bias is irrelevant here: bound
        is at most a few hundred against a 2**32 range, and the requirement is
        reproducibility, not cryptographic uniformity."""
        if bound < 1:
            raise ValueError("BLOCKED_SEED:bound_must_be_positive")
        return self.next_u32() % bound

    @property
    def draws(self) -> int:
        return self._draws


def durstenfeld(n: int, seed: int) -> list[int]:
    """Permutation of range(n) by Durstenfeld shuffle driven by `Lcg`.

    for i = n-1 .. 1:  j = next_u32() mod (i+1);  swap a[i], a[j]

    Worked example (n=4, seed=7), verifiable by longhand arithmetic:
        x1 = (1664525*7 + 1013904223) mod 2**32 = 1025555898
        j  = 1025555898 mod 4 = 2      -> swap a[3],a[2]: [0,1,3,2]
    Later steps follow the same recurrence; see the test suite, which re-derives
    the whole sequence from the recurrence independently of this function.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("BLOCKED_SEED:n_must_be_positive_int")
    rng = Lcg(seed)
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.below(i + 1)
        order[i], order[j] = order[j], order[i]
    return order


# --------------------------------------------------------------------------- #
# integrity helpers
# --------------------------------------------------------------------------- #

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def png_bytes(image: Image.Image) -> bytes:
    """Lossless, deterministic serialisation.

    ALL arms are serialised through this function -- including the unperturbed
    control -- so that arm differences cannot be an artefact of JPEG vs PNG
    encoding. The control is the decoded original re-encoded as PNG, never the
    original JPEG file bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise ValueError(f"BLOCKED_IMAGING:missing_image:{path}")
    with Image.open(path) as handle:
        return handle.convert("RGB")


def histogram_l1(left: Image.Image, right: Image.Image) -> int:
    """Sum |h_left - h_right| over all channel bins. 0 iff the pixel multisets
    agree channel-wise -- the operational test of multiset preservation."""
    if left.size != right.size or left.mode != right.mode:
        raise ValueError("BLOCKED_IMAGING:histogram_shape_mismatch")
    a, b = left.histogram(), right.histogram()
    return sum(abs(x - y) for x, y in zip(a, b))


def resolve_bbox(bbox: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Normalised [x0,y0,x1,y1] in [0,1] -> integer pixel box (left, top, right, bottom).

    Raises on anything that does not land strictly inside the image: an item that
    declares a bbox outside its own page is a construction defect and must fail
    loudly rather than be silently clipped to something measurable.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("BLOCKED_BBOX:bbox_must_be_4_numbers")
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("BLOCKED_BBOX:bbox_must_be_numeric")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if not (0.0 <= x0 < x1 <= 1.0) or not (0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"BLOCKED_BBOX:out_of_unit_square_or_inverted:{list(bbox)}")
    left, top = int(round(x0 * width)), int(round(y0 * height))
    right, bottom = int(round(x1 * width)), int(round(y1 * height))
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(f"BLOCKED_BBOX:out_of_bounds:{list(bbox)}:{width}x{height}")
    if right - left < 1 or bottom - top < 1:
        raise ValueError(f"BLOCKED_BBOX:degenerate_after_rounding:{list(bbox)}")
    return left, top, right, bottom


@dataclass
class Perturbation:
    """One counterfactual image plus everything needed to replay and audit it."""

    family: str
    image: Image.Image
    preserves_multiset: bool
    histogram_l1: int
    is_identity: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def audit(self) -> dict[str, Any]:
        payload = {
            "family": self.family,
            "preserves_multiset": self.preserves_multiset,
            "histogram_l1": self.histogram_l1,
            "is_identity": self.is_identity,
            "size": list(self.image.size),
        }
        payload.update(self.detail)
        return payload


# --------------------------------------------------------------------------- #
# P2 BLOCK_SHUFFLE
# --------------------------------------------------------------------------- #

def block_grid(width: int, height: int, k: int) -> tuple[int, int, int, int]:
    """Largest centred region divisible by k. Returns (off_x, off_y, bw, bh).

    The border strip that does not fit the grid is left UNTOUCHED rather than
    cropped away, because cropping would delete pixels and break the multiset
    guarantee that this whole design rests on.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("BLOCKED_GRID:k_must_be_int_ge_2")
    bw, bh = width // k, height // k
    if bw < 1 or bh < 1:
        raise ValueError(f"BLOCKED_GRID:image_too_small_for_k:{width}x{height}:k={k}")
    used_w, used_h = bw * k, bh * k
    return (width - used_w) // 2, (height - used_h) // 2, bw, bh


def block_shuffle(image: Image.Image, k: int = 3, seed: int = 0,
                  permutation: Sequence[int] | None = None) -> Perturbation:
    """P2. Permute the k*k equal interior blocks.

    `permutation` overrides the seeded order; the identity permutation therefore
    yields an output whose pixels -- and whose PNG bytes -- are identical to the
    input. That is the control used to prove the pipeline itself introduces no
    difference between arms.
    """
    width, height = image.size
    off_x, off_y, bw, bh = block_grid(width, height, k)
    n = k * k

    if permutation is None:
        order = durstenfeld(n, seed)
        source = "lcg_durstenfeld"
    else:
        order = list(permutation)
        if sorted(order) != list(range(n)):
            raise ValueError(f"BLOCKED_GRID:permutation_not_a_bijection_of_{n}")
        source = "explicit"

    out = image.copy()
    tiles = []
    for index in range(n):
        row, col = divmod(index, k)
        box = (off_x + col * bw, off_y + row * bh,
               off_x + (col + 1) * bw, off_y + (row + 1) * bh)
        tiles.append(image.crop(box))
    for index in range(n):
        row, col = divmod(index, k)
        box = (off_x + col * bw, off_y + row * bh,
               off_x + (col + 1) * bw, off_y + (row + 1) * bh)
        out.paste(tiles[order[index]], box)

    identity = order == list(range(n))
    # Interior seams introduced: each adjacent block pair whose source blocks
    # were not adjacent in the original. Upper bound on OCR token splits.
    seams = 2 * k * (k - 1)
    return Perturbation(
        family=FAMILY_P2,
        image=out,
        preserves_multiset=True,
        histogram_l1=histogram_l1(image, out),
        is_identity=identity,
        detail={
            "k": k, "seed": seed, "permutation": order, "permutation_source": source,
            "block_w": bw, "block_h": bh, "grid_offset": [off_x, off_y],
            "untouched_border_px": [width - bw * k, height - bh * k],
            "seam_count": seams,
        },
    )


# --------------------------------------------------------------------------- #
# P1 LABEL_PERMUTE
# --------------------------------------------------------------------------- #

def otsu_threshold(gray: Image.Image) -> int:
    """Otsu's method on the 256-bin grey histogram. Pure arithmetic, no numpy."""
    hist = gray.histogram()
    total = sum(hist)
    if total == 0:
        raise ValueError("BLOCKED_IMAGING:empty_image")
    sum_all = sum(i * hist[i] for i in range(256))
    w_b = 0
    sum_b = 0
    best_var, best_t = -1.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (mean_b - mean_f) ** 2
        if between > best_var:
            best_var, best_t = between, t
    return best_t


def detect_text_lines(image: Image.Image, *, min_height: int = 8,
                      min_width: int = 20, merge_gap: int = 2,
                      ink_row_frac: float = 0.005) -> list[tuple[int, int, int, int]]:
    """Horizontal-projection line detection. Returns [(left, top, right, bottom)].

    Deliberately simple and dependency-free: we do not need reading order or
    semantic labels, only equal-sized swappable regions that contain whole glyph
    rows. A missed line costs recall (fewer candidate pairs), never correctness.
    """
    gray = image.convert("L")
    threshold = otsu_threshold(gray)
    width, height = gray.size
    # Flat byte buffer indexed as buf[y * width + x]. Using tobytes() instead of
    # Image.load() keeps this pure-arithmetic and roughly two orders of magnitude
    # faster than per-pixel PixelAccess calls on a 1094x1602 page.
    buf = gray.tobytes()

    row_ink: list[int] = []
    for y in range(height):
        row = buf[y * width:(y + 1) * width]
        count = 0
        for value in row:
            if value <= threshold:
                count += 1
        row_ink.append(count)

    min_ink = max(1, int(ink_row_frac * width))
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for y in range(height):
        if row_ink[y] >= min_ink:
            if start is None:
                start = y
        else:
            if start is not None:
                spans.append((start, y))
                start = None
    if start is not None:
        spans.append((start, height))

    merged: list[list[int]] = []
    for top, bottom in spans:
        if merged and top - merged[-1][1] <= merge_gap:
            merged[-1][1] = bottom
        else:
            merged.append([top, bottom])

    lines: list[tuple[int, int, int, int]] = []
    for top, bottom in merged:
        if bottom - top < min_height:
            continue
        col_ink = bytearray(width)
        for y in range(top, bottom):
            row = buf[y * width:(y + 1) * width]
            for x, value in enumerate(row):
                if value <= threshold:
                    col_ink[x] = 1
        left: int | None = None
        right: int | None = None
        for x in range(width):
            if col_ink[x]:
                if left is None:
                    left = x
                right = x + 1
        if left is None or right is None or right - left < min_width:
            continue
        lines.append((left, top, right, bottom))
    return lines


def _expand_to(box: tuple[int, int, int, int], target_w: int, target_h: int,
               width: int, height: int) -> tuple[int, int, int, int] | None:
    """Grow `box` symmetrically to exactly (target_w, target_h) inside the image.

    Returns None if it cannot fit. Expansion (rather than cropping to the smaller
    box) is what keeps the swap a pure pixel permutation: two disjoint regions of
    identical shape can be exchanged without creating or destroying a pixel.
    """
    left, top, right, bottom = box
    if target_w > width or target_h > height:
        return None
    grow_x = target_w - (right - left)
    grow_y = target_h - (bottom - top)
    left -= grow_x // 2
    right += grow_x - grow_x // 2
    top -= grow_y // 2
    bottom += grow_y - grow_y // 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    if left < 0 or top < 0 or right > width or bottom > height:
        return None
    return left, top, right, bottom


def _disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def label_permute(image: Image.Image, seed: int = 0, *,
                  lines: Sequence[tuple[int, int, int, int]] | None = None,
                  max_pairs: int = 400) -> Perturbation:
    """P1. Swap two equal-sized, disjoint detected line regions.

    Raises BLOCKED_P1 when the page yields no admissible pair; a page with one
    text line has no position-to-label binding to destroy, and silently returning
    the original would fabricate an NDE of exactly 0.
    """
    width, height = image.size
    boxes = list(lines) if lines is not None else detect_text_lines(image)
    if len(boxes) < 2:
        raise ValueError(f"BLOCKED_P1:need_2_text_lines_found_{len(boxes)}")

    candidates: list[tuple[int, int]] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            candidates.append((i, j))
    order = durstenfeld(len(candidates), seed)

    chosen = None
    for slot in order[:max_pairs]:
        i, j = candidates[slot]
        a, b = boxes[i], boxes[j]
        target_w = max(a[2] - a[0], b[2] - b[0])
        target_h = max(a[3] - a[1], b[3] - b[1])
        ea = _expand_to(a, target_w, target_h, width, height)
        eb = _expand_to(b, target_w, target_h, width, height)
        if ea is None or eb is None or ea == eb or not _disjoint(ea, eb):
            continue
        chosen = (i, j, ea, eb, target_w, target_h)
        break
    if chosen is None:
        raise ValueError("BLOCKED_P1:no_disjoint_equal_sized_pair")

    i, j, ea, eb, target_w, target_h = chosen
    out = image.copy()
    tile_a = image.crop(ea)
    tile_b = image.crop(eb)
    out.paste(tile_b, ea)
    out.paste(tile_a, eb)

    return Perturbation(
        family=FAMILY_P1,
        image=out,
        preserves_multiset=True,
        histogram_l1=histogram_l1(image, out),
        is_identity=False,
        detail={
            "seed": seed,
            "n_lines_detected": len(boxes),
            "swapped_line_indices": [i, j],
            "swapped_boxes_px": [list(ea), list(eb)],
            "region_w": target_w, "region_h": target_h,
            "note": "regions expanded to a common shape so the swap is a pure "
                    "pixel permutation; expansion may include adjacent "
                    "whitespace, which does not affect the token multiset",
        },
    )


# --------------------------------------------------------------------------- #
# P3 REGION_OCCLUDE (necessity probe -- NOT an NDE arm)
# --------------------------------------------------------------------------- #

def region_occlude(image: Image.Image, bbox: Sequence[float]) -> Perturbation:
    """P3. Black out exactly the declared bbox. Deletes content by design."""
    width, height = image.size
    box = resolve_bbox(bbox, width, height)
    out = image.copy()
    out.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (0, 0, 0)), box)
    return Perturbation(
        family=FAMILY_P3,
        image=out,
        preserves_multiset=False,
        histogram_l1=histogram_l1(image, out),
        is_identity=False,
        detail={
            "bbox_norm": [float(v) for v in bbox],
            "bbox_px": list(box),
            "note": "targeted necessity probe; deletes content, so it is NOT a "
                    "structure-preserving counterfactual and must not be pooled "
                    "into NDE_I",
        },
    )


def build_arms(path: Path, *, k: int = 3, seed: int = 0,
               bbox: Sequence[float] | None = None) -> dict[str, dict[str, Any]]:
    """Serialise control + P1 + P2 (+ P3 when a bbox is declared) to PNG bytes.

    The control is the decoded original re-encoded as PNG so that every arm goes
    through the identical encoder.
    """
    original = load_rgb(path)
    control_png = png_bytes(original)
    arms: dict[str, dict[str, Any]] = {
        "control": {
            "png": control_png,
            "audit": {"family": "CONTROL_PNG", "preserves_multiset": True,
                      "histogram_l1": 0, "is_identity": True,
                      "size": list(original.size),
                      "source_path": str(path),
                      "source_sha256": sha256_bytes(path.read_bytes()),
                      "png_sha256": sha256_bytes(control_png)},
        }
    }
    for family, perturbation in (
        (FAMILY_P1, label_permute(original, seed=seed)),
        (FAMILY_P2, block_shuffle(original, k=k, seed=seed)),
    ):
        data = png_bytes(perturbation.image)
        audit = perturbation.audit()
        audit["png_sha256"] = sha256_bytes(data)
        arms[family] = {"png": data, "audit": audit}
    if bbox is not None:
        perturbation = region_occlude(original, bbox)
        data = png_bytes(perturbation.image)
        audit = perturbation.audit()
        audit["png_sha256"] = sha256_bytes(data)
        arms[FAMILY_P3] = {"png": data, "audit": audit}
    return arms


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build counterfactual page images.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bbox", type=float, nargs=4, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    arms = build_arms(args.image, k=args.k, seed=args.seed, bbox=args.bbox)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, arm in arms.items():
            (args.out_dir / f"{args.image.stem}__{name}.png").write_bytes(arm["png"])
    print(json.dumps({"schema": SCHEMA,
                      "arms": {n: a["audit"] for n, a in arms.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
