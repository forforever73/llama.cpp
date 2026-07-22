#pragma once

#include "ggml-metal-device.h"
#include "ggml.h"

#include <cstdint>

namespace ggml_metal_tuning {

// Bucket edges partition the (K, N0, tokens) shape space into cells; one tuned
// row per cell holds the winning tile. Edges are DEVICE-INDEPENDENT: they track
// tile geometry (NR0/NR1 granularity) and real model-dimension clusters, not any
// single GPU's crossover point. Which tile wins inside a cell is per-device data
// held in mm_tile_tuned_table, so a device's flip point is never baked in here.
//
// K (in-feat, ne00): 0=<2048, 1=2048-8191, 2=8192-16383, 3=>=16384
// (edge at 8192 separates the 4096/5120 hidden-dim cluster from large-K FFN/attn)
constexpr int MM_TILE_K_BUCKETS[]      = { 2048, 8192, 16384 };
// out-feat (ne01): 0=<2048, 1=2048-8191, 2=8192-29999, 3=>=30000 (vocab isolated)
constexpr int MM_TILE_N0_BUCKETS[]     = { 2048, 8192, 30000 };
// token (ne11): 0=<32, 1=32-63, 2=64-127, 3=128-255, 4=>=256. Split finely across
// the low-token range: the small-tile vs baseline crossover sits here and drifts
// per (dtype,K,N0), so each side of it gets its own bucket to key a tuned row.
constexpr int MM_TILE_TOKEN_BUCKETS[]  = { 32, 64, 128, 256 };

int mm_tile_K_bucket(int64_t K);
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
constexpr mm_tile_cfg_t MM_TILE_FAMILY[] = {
    { 32, 8 }, { 32, 16 }, { 64, 8 }, { 64, 16 }, { 128, 16 }, { 128, 32 },
};
constexpr mm_tile_cfg_t MM_TILE_BASELINE_CFG = { 64, 32 };

// Tuned table has two row kinds.
// Exact rows key a (K_b, N0_b, tokens_b) bucket.
// Default rows collapse the high-order dims but ALWAYS keep tokens_b, per the
// pick lattice (folding order = weakest signal first, tokens never folds):
//   L1 exact -> L2a (K_b=ANY, keep N0_b, keep tokens_b)
//            -> L2b (K_b=ANY, N0_b=ANY, keep tokens_b) -> L3 baseline (64x32).
// tokens is the strongest tile signal (small tile at low tokens, large tile at
// prefill), so it is never collapsed. MM_TILE_BUCKET_ANY is the collapse
// sentinel; it applies to K_b or N0_b and is compared per-field.
constexpr int8_t MM_TILE_BUCKET_ANY    = -1;

struct mm_tile_key_t {
    int8_t  device_id;
    int8_t  dtype;      // ggml_type of src0
    int8_t  K_b;        // in-feat (ne00) bucket, or MM_TILE_BUCKET_ANY when collapsed
    int8_t  N0_b;       // out-feat (ne01) bucket, or MM_TILE_BUCKET_ANY when collapsed
    int8_t  tokens_b;   // token (ne11) bucket (never collapsed)
    int8_t  _pad[3];
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
                             int64_t K,
                             int64_t N_out,
                             int64_t tokens);

// Self-test for the pick lattice (L1 exact -> L2a K-collapse keeping N0 ->
// L2b K+N0 collapse -> L3 baseline). Runs the real lookup against a synthetic
// table; returns the number of failed assertions (0 = pass). Wired into `tune`.
int            mm_tile_lattice_selftest();

}  // namespace ggml_metal_tuning
