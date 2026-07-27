#pragma once

#include "ggml-metal-device.h"
#include "ggml.h"

#include <cstdint>

namespace ggml_metal_tuning {

// Bucket edges partition the (N0, tokens) shape space into cells; one tuned
// row per cell holds the winning tile. Edges are DEVICE-INDEPENDENT: they track
// tile geometry (NR0/NR1 granularity) and real model-dimension clusters, not any
// single GPU's crossover point. Which tile wins inside a cell is per-device data
// held in mm_tile_tuned_table, so a device's flip point is never baked in here.
//
// K (in-feat, ne00) is NOT a key dimension: it scales every candidate's cost
// equally, so it never changes the tile ranking, and keying on it lets exact
// rows memorize deep-K quirks that then misroute unsampled neighbor shapes.
// The sweep still samples deep-K shapes; group_split aggregates across K.
//
// out-feat (ne01): 0=<2048, 1=2048-4607, 2=4608-8191, 3=8192-29999, 4=>=30000 (vocab isolated).
// Edge at 4608 splits the 3584/4096 cluster from 4608/5120: the crossover drifts with N0
// (baseline threadgroup count -> occupancy), so 5120 must not inherit the 3584/4096 winner.
constexpr int MM_TILE_N0_BUCKETS[]     = { 2048, 4608, 8192, 30000 };
// token (ne11): 0=<32, 1=32-63, 2=64-127, 3=>=128. Split finely across the
// low-token range: the small-tile vs baseline crossover sits here and drifts
// per (dtype,K,N0), so each side of it gets its own bucket to key a tuned row.
// There is no bucket above 128: mm_tile_pick short-circuits tokens >= 256 to
// baseline (every tuned device converged to baseline there), so bucket-3 rows
// effectively serve [128,256). Keep that short-circuit in sync with this edge.
constexpr int MM_TILE_TOKEN_BUCKETS[]  = { 32, 64, 128 };

int mm_tile_N0_bucket(int64_t N_out);
int mm_tile_token_bucket(int64_t tokens);

// default cfg used when no tuned row matches
struct mm_tile_cfg_t {
    int16_t nr0;  // 32, 64, or 128 (int16: 128 exceeds int8_t range)
    int16_t nr1;  //  8, 16, or  32
};

// Single source of truth for the instantiated tile variants in mul_mm.metal.
// Tuned-table rows are static_assert'd to be a family member or the baseline,
// and run_mm_tile_tune_check exercises every member. 64x32 is NOT a family
// member; it is the baseline, served by the bare kernel_mul_mm.
// 128x16/128x32 were instantiated during tuning but no tuned row on any device
// picked them (metallib/compile-time dead weight); re-add here, in
// mul_mm.metal, and in mm_tile_legal_configs when retuning a new device.
constexpr mm_tile_cfg_t MM_TILE_FAMILY[] = {
    { 32, 8 }, { 32, 16 }, { 64, 8 }, { 64, 16 },
};
constexpr mm_tile_cfg_t MM_TILE_BASELINE_CFG = { 64, 32 };

// Occupancy-saturation threshold for the pick-time veto, measured in baseline
// (64x32) threadgroup count: n_tg = ceil(N_out/64) * ceil(tokens/32). Once the
// baseline dispatch alone saturates the GPU (n_tg >= C_sat), a smaller tile has
// no occupancy headroom left to win, so any non-baseline table row is overridden
// back to baseline. This bounds table extrapolation error on shapes the sweep
// never sampled. tokens < 32 is exempt: those wins come from baseline's token
// padding + bc_out slow path, which persist at any occupancy.
// Calibrated on M4 Max (40 cores x ~3.6 concurrent tg/core; accuracy is flat
// for C_sat in [128,160]). Per-device data like the tuned table: recalibrate
// when adding rows for a new device. Overestimating C_sat degrades to pure
// table behavior; underestimating only forfeits small-tile wins - neither
// direction can pick something slower than baseline.
constexpr int MM_TILE_C_SAT_M4_MAX = 144;

// Tuned table has two row kinds.
// Exact rows key a (N0_b, tokens_b) bucket.
// Default rows collapse N0_b but ALWAYS keep tokens_b, per the pick lattice
// (folding order = weakest signal first, tokens never folds):
//   L1 exact -> L2 (N0_b=ANY, keep tokens_b) -> L3 baseline (64x32).
// tokens is the strongest tile signal (small tile at low tokens, large tile at
// prefill), so it is never collapsed. MM_TILE_BUCKET_ANY is the collapse
// sentinel; it applies to N0_b and is compared per-field.
constexpr int8_t MM_TILE_BUCKET_ANY    = -1;

struct mm_tile_key_t {
    int8_t  device_id;
    int8_t  dtype;      // ggml_type of src0
    int8_t  N0_b;       // out-feat (ne01) bucket, or MM_TILE_BUCKET_ANY when collapsed
    int8_t  tokens_b;   // token (ne11) bucket (never collapsed)
    int8_t  _pad[4];
};
static_assert(sizeof(mm_tile_key_t) == 8, "mm_tile_key_t must be 8 bytes for memcmp");

struct mm_tile_entry_t {
    mm_tile_key_t key;
    mm_tile_cfg_t cfg;
};

// test/tune-only override; when set, mm_tile_pick returns it directly.
void           mm_tile_set_override(mm_tile_cfg_t cfg);
void           mm_tile_clear_override();
bool           mm_tile_override_active();
mm_tile_cfg_t  mm_tile_baseline_cfg();

// Returns (64,32) unless a tuned row matches.
mm_tile_cfg_t  mm_tile_pick(enum ggml_metal_device_id device_id,
                             int dtype,
                             int64_t N_out,
                             int64_t tokens);

// Self-test for the pick lattice (L1 exact -> L2 N0-collapse -> L3 baseline).
// Runs the real lookup against a synthetic table; returns the number of failed
// assertions (0 = pass). Wired into `tune`.
int            mm_tile_lattice_selftest();

}  // namespace ggml_metal_tuning
