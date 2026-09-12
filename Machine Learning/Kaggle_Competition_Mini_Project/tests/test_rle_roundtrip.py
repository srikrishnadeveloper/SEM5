"""
RLE round-trip tests as mandated by Grok:
1. Nonzero mask: encode → decode → assert pixel-identical
2. All-zero 2048×2048 mask: encode → counts is non-empty string → decode → sum == 0
3. Verify that we NEVER hardcode RLE strings — always call pycocotools
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from metrics.pq import encode_mask, decode_rle


def test_nonzero_roundtrip():
    """Encode a blob mask → decode → pixel-identical."""
    h, w = 512, 512
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[100:300, 150:400] = 1
    assert mask.sum() > 0, "Mask must be nonzero"

    counts = encode_mask(mask)
    assert isinstance(counts, str), f"Expected str, got {type(counts)}"
    assert len(counts) > 0, "Counts string must be non-empty"

    decoded = decode_rle(counts, h, w)
    assert decoded.shape == (h, w), f"Shape mismatch: {decoded.shape}"
    assert np.array_equal(mask, decoded), "Round-trip failed: decoded != original"
    print(f"[PASS] test_nonzero_roundtrip: {mask.sum()} pixels, counts='{counts[:40]}...'")


def test_zero_2048_roundtrip():
    """All-zero 2048×2048 mask → encode → non-empty counts → decode → sum == 0.

    Grok mandate: never hardcode 'PPPP4'. Always encode through pycocotools.
    Whatever pycocotools produces is the canonical encoding.
    """
    h, w = 2048, 2048
    mask = np.zeros((h, w), dtype=np.uint8)

    counts = encode_mask(mask)
    assert isinstance(counts, str), f"Expected str, got {type(counts)}"
    assert len(counts) > 0, "Counts string must be non-empty for zero mask"

    decoded = decode_rle(counts, h, w)
    assert decoded.shape == (h, w), f"Shape mismatch: {decoded.shape}"
    assert decoded.sum() == 0, f"Decoded zero mask has sum={decoded.sum()}"

    print(f"[PASS] test_zero_2048_roundtrip: counts='{counts}', decoded sum=0")


def test_varied_sizes():
    """Round-trip on several mask sizes to ensure generality."""
    for h, w in [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]:
        mask = np.zeros((h, w), dtype=np.uint8)
        # Put a small blob
        mask[h // 4 : h // 2, w // 4 : w // 2] = 1
        counts = encode_mask(mask)
        decoded = decode_rle(counts, h, w)
        assert np.array_equal(mask, decoded), f"Round-trip failed at {h}×{w}"
    print("[PASS] test_varied_sizes: 256, 512, 1024, 2048 all pass")


def test_no_hardcoded_strings():
    """Ensure encode_mask does not return any known hardcoded dummy string.

    This test documents that 'PPP2' (the old dummy) is never produced
    by pycocotools for any real mask.
    """
    h, w = 2048, 2048

    # Zero mask
    zero_mask = np.zeros((h, w), dtype=np.uint8)
    counts_zero = encode_mask(zero_mask)
    assert counts_zero != "PPP2", "Got dummy 'PPP2' — hardcoded!"

    # Nonzero mask
    nonzero_mask = np.zeros((h, w), dtype=np.uint8)
    nonzero_mask[500:1500, 500:1500] = 1
    counts_nonzero = encode_mask(nonzero_mask)
    assert counts_nonzero != "PPP2", "Got dummy 'PPP2' — hardcoded!"

    print(f"[PASS] test_no_hardcoded_strings: zero->'{counts_zero}', nonzero->'{counts_nonzero[:30]}...'")


if __name__ == "__main__":
    test_nonzero_roundtrip()
    test_zero_2048_roundtrip()
    test_varied_sizes()
    test_no_hardcoded_strings()
    print("\n=== ALL 4 RLE ROUND-TRIP TESTS PASSED ===")
