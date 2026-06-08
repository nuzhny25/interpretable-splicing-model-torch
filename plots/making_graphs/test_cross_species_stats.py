"""One-shot sanity check for cross_species_stats. Run, eyeball, delete."""
import numpy as np

# Function copied inline so this test doesn't run plot_combined_sr.py's
# top-level data loading (which needs real .npz / .json files on disk).
STEP = 10


def cross_species_stats(species_dict, step=STEP):
    all_pos = np.concatenate([pos for pos, _ in species_dict.values()])
    grid = np.arange(all_pos.min(), all_pos.max() + 1, step)
    tol = step / 2
    stacked = []
    for pos, vals in species_dict.values():
        order = np.argsort(pos)
        pos_s = pos[order]
        vals_s = vals[order].astype(float)
        ins = np.searchsorted(pos_s, grid)
        left = np.clip(ins - 1, 0, len(pos_s) - 1)
        right = np.clip(ins, 0, len(pos_s) - 1)
        dist_left = np.abs(grid - pos_s[left])
        dist_right = np.abs(grid - pos_s[right])
        use_right = dist_right < dist_left
        nearest_idx = np.where(use_right, right, left)
        nearest_dist = np.where(use_right, dist_right, dist_left)
        col = np.where(nearest_dist <= tol, vals_s[nearest_idx], np.nan)
        stacked.append(col)
    stacked = np.array(stacked)
    valid = np.sum(~np.isnan(stacked), axis=0) >= 2
    med = np.nanmedian(stacked, axis=0)
    q25 = np.nanpercentile(stacked, 25, axis=0)
    q75 = np.nanpercentile(stacked, 75, axis=0)
    sd = np.nanstd(stacked, axis=0)
    for arr in (med, q25, q75, sd):
        arr[~valid] = np.nan
    return grid, med, q25, q75, sd


def approx_equal(a, b, atol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    nan_mask = np.isnan(a) & np.isnan(b)
    return np.all(nan_mask | (np.abs(a - b) <= atol))


# --- Test 1: median / IQR / SD math on aligned grids ---
# Four species, identical positions, values [0, 1, 2, 3] at each grid column.
# Expect: median = 1.5, q25 = 0.75, q75 = 2.25, sd = std([0,1,2,3]) ≈ 1.118
positions = np.array([0, 10, 20, 30])
species = {
    "a": (positions, np.zeros(4)),
    "b": (positions, np.ones(4)),
    "c": (positions, np.full(4, 2.0)),
    "d": (positions, np.full(4, 3.0)),
}
grid, med, q25, q75, sd = cross_species_stats(species)
expected_sd = np.std([0, 1, 2, 3])  # population SD, matches np.nanstd default
print("Test 1: aligned grids with values [0,1,2,3] per column")
print(f"  grid       = {grid}")
print(f"  median     = {med}        expect [1.5, 1.5, 1.5, 1.5]")
print(f"  q25        = {q25}        expect [0.75, 0.75, 0.75, 0.75]")
print(f"  q75        = {q75}        expect [2.25, 2.25, 2.25, 2.25]")
print(f"  sd         = {sd}  expect ~{expected_sd:.4f} each")
assert approx_equal(grid, [0, 10, 20, 30])
assert approx_equal(med, [1.5] * 4)
assert approx_equal(q25, [0.75] * 4)
assert approx_equal(q75, [2.25] * 4)
assert approx_equal(sd, [expected_sd] * 4)
print("  PASS\n")

# --- Test 2: step/2 tolerance dropout ---
# Species c has only two positions, 0 and 100. The grid spans 0..100 in
# steps of 10. At grid columns 10..90, c's nearest position is at least 10
# away (> tol=5), so c should NOT contribute. At columns 0 and 100 it does.
species = {
    "a": (np.arange(0, 101, 10), np.ones(11)),
    "b": (np.arange(0, 101, 10), np.full(11, 3.0)),
    "c": (np.array([0, 100]), np.array([99.0, 99.0])),
}
grid, med, q25, q75, sd = cross_species_stats(species)
print("Test 2: species 'c' only present at the endpoints — interior must exclude it")
print(f"  grid   = {grid}")
print(f"  median = {med}")
# Endpoints include c (median of [1,3,99] = 3); interior is median of [1,3] = 2.
expected_med = [3.0] + [2.0] * 9 + [3.0]
print(f"  expect {expected_med}")
assert approx_equal(med, expected_med)
# Interior SD = std([1, 3]) = 1; endpoints = std([1, 3, 99]) ≈ 46.20
expected_sd_interior = 1.0
expected_sd_endpoints = np.std([1, 3, 99])
print(f"  sd     = {sd}")
print(f"  expect endpoints ~{expected_sd_endpoints:.2f}, interior {expected_sd_interior}")
assert approx_equal(sd[1:-1], [expected_sd_interior] * 9)
assert approx_equal([sd[0], sd[-1]], [expected_sd_endpoints, expected_sd_endpoints])
print("  PASS\n")

# --- Test 2b: shift by 5 (== tol) should INCLUDE the species ---
species_inclusive = {
    "a": (np.array([0, 10, 20, 30]), np.array([1.0, 1.0, 1.0, 1.0])),
    "b": (np.array([0, 10, 20, 30]), np.array([3.0, 3.0, 3.0, 3.0])),
    "c": (np.array([5, 15, 25, 35]), np.array([100.0, 100.0, 100.0, 100.0])),
}
grid, med, q25, q75, sd = cross_species_stats(species_inclusive)
print("Test 2b: species 'c' shifted by 5 (== tol), should be INCLUDED")
print(f"  median = {med}  expect [3, 3, 3, 3]  (median of [1, 3, 100])")
assert approx_equal(med, [3.0, 3.0, 3.0, 3.0])
print("  PASS\n")

# --- Test 3: n >= 2 mask — single-contributor columns become NaN ---
# Grid spans 0..30. Only species 'a' covers position 30; everyone else stops at 20.
species = {
    "a": (np.array([0, 10, 20, 30]), np.array([1.0, 1.0, 1.0, 42.0])),
    "b": (np.array([0, 10, 20]), np.array([2.0, 2.0, 2.0])),
    "c": (np.array([0, 10, 20]), np.array([3.0, 3.0, 3.0])),
}
grid, med, q25, q75, sd = cross_species_stats(species)
print("Test 3: only species 'a' covers grid column 30 → column should be NaN everywhere")
print(f"  grid   = {grid}")
print(f"  median = {med}   expect [2, 2, 2, nan]")
print(f"  q25    = {q25}   expect [1.5, 1.5, 1.5, nan]")
print(f"  q75    = {q75}   expect [2.5, 2.5, 2.5, nan]")
print(f"  sd     = {sd}   last entry should be nan")
assert approx_equal(med, [2.0, 2.0, 2.0, np.nan])
assert approx_equal(q25, [1.5, 1.5, 1.5, np.nan])
assert approx_equal(q75, [2.5, 2.5, 2.5, np.nan])
assert np.isnan(sd[-1]) and not np.any(np.isnan(sd[:-1]))
print("  PASS\n")

# --- Test 4: gap in one species' positions should NaN only nearby columns ---
# Species 'a' has a big gap between 10 and 50; grid columns 20, 30, 40 are
# all > tol=5 away from {10, 50}, so 'a' should drop out there.
species = {
    "a": (np.array([0, 10, 50, 60]), np.array([1.0, 1.0, 1.0, 1.0])),
    "b": (np.arange(0, 61, 10), np.full(7, 2.0)),
    "c": (np.arange(0, 61, 10), np.full(7, 3.0)),
}
grid, med, q25, q75, sd = cross_species_stats(species)
print("Test 4: species 'a' has a gap; columns 20/30/40 use only b,c → median 2.5")
print(f"  grid   = {grid}")
print(f"  median = {med}")
print("  expect [2, 2, 2.5, 2.5, 2.5, 2, 2]")
assert approx_equal(med, [2.0, 2.0, 2.5, 2.5, 2.5, 2.0, 2.0])
print("  PASS\n")

print("All tests passed.")
