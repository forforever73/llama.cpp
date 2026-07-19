#pragma once

#include "ggml-metal-device.h"
#include "ggml.h"

#include <cstdint>

namespace ggml_metal_tuning {

// K-dimension buckets: 0=short(<4096), 1=mid(4096-16383), 2=long(>=16384)
constexpr int MM_TILE_K_BUCKETS[]      = { 4096, 16384 };
// token (ne11) buckets: 0=decode(<=8), 1=small(9-32), 2=medium+(>=33)
constexpr int MM_TILE_TOKEN_BUCKETS[]  = { 9, 33 };

int mm_tile_K_bucket(int64_t K);
int mm_tile_token_bucket(int64_t tokens);

// default cfg used when no tuned row matches
struct mm_tile_cfg_t {
    int16_t nr0;  // 32, 64, or 128 (int16: 128 exceeds int8_t range)
    int16_t nr1;  //  8, 16, or  32
};

// Tuned table has two row kinds.
// Exact rows key a (K_b, tokens_b) bucket.
// Default rows collapse K over one tokens domain:
//   K_b == MM_TILE_NR_DEFAULT, tokens_b == domain id.
// mm_tile_pick: exact bucket -> domain default -> baseline (64x32).
constexpr int8_t MM_TILE_NR_DEFAULT    = -1;
constexpr int8_t MM_TILE_DOMAIN_DECODE =  0;  // tokens <= 8
constexpr int8_t MM_TILE_DOMAIN_BATCH  =  1;  // tokens >= 9

struct mm_tile_key_t {
    int8_t  device_id;
    int8_t  dtype;      // ggml_type of src0
    int8_t  K_b;        // K bucket (0/1/2), or MM_TILE_NR_DEFAULT for domain-default
    int8_t  tokens_b;   // token bucket or domain id
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
                             int gpu_family,
                             int dtype,
                             int64_t K,
                             int64_t tokens);

}  // namespace ggml_metal_tuning
