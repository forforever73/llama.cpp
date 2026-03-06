#include "llama-batch.h"
#include "llama-memory.h"
#include "llama-vocab.h"
#include "ggml.h"

#include <map>
#include <vector>

namespace {

struct dummy_memory final : llama_memory_i {
    std::vector<llama_pos> min_pos;
    std::vector<llama_pos> max_pos;

    dummy_memory() : min_pos(LLAMA_MAX_SEQ, -1), max_pos(LLAMA_MAX_SEQ, -1) {}

    void set_seq(llama_seq_id id, llama_pos min_v, llama_pos max_v) {
        min_pos[id] = min_v;
        max_pos[id] = max_v;
    }

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & /*balloc*/,
            uint32_t /*n_ubatch*/,
            bool /*embd_all*/) override {
        return nullptr;
    }

    llama_memory_context_ptr init_full() override {
        return nullptr;
    }

    llama_memory_context_ptr init_update(llama_context * /*lctx*/, bool /*optimize*/) override {
        return nullptr;
    }

    bool get_can_shift() const override {
        return false;
    }

    void clear(bool /*data*/) override {}

    bool seq_rm(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/) override {
        return false;
    }

    void seq_cp(llama_seq_id /*seq_id_src*/, llama_seq_id /*seq_id_dst*/, llama_pos /*p0*/, llama_pos /*p1*/) override {}

    void seq_keep(llama_seq_id /*seq_id*/) override {}

    void seq_add(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/, llama_pos /*shift*/) override {}

    void seq_div(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/, int /*d*/) override {}

    llama_pos seq_pos_min(llama_seq_id seq_id) const override {
        return min_pos[seq_id];
    }

    llama_pos seq_pos_max(llama_seq_id seq_id) const override {
        return max_pos[seq_id];
    }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override {
        return {};
    }

    void state_write(llama_io_write_i & /*io*/, llama_seq_id /*seq_id*/, llama_state_seq_flags /*flags*/) const override {}

    void state_read(llama_io_read_i & /*io*/, llama_seq_id /*seq_id*/, llama_state_seq_flags /*flags*/) override {}
};

} // namespace

int main() {
    llama_vocab vocab;
    dummy_memory memory;
    memory.set_seq(1, 0, 19);

    llama_seq_id seq_id_val = 1;
    llama_seq_id * seq_id_ptrs[1] = { &seq_id_val };
    int32_t n_seq_id = 1;
    llama_pos pos = 21;
    float embd[1] = { 0.0f };

    llama_batch batch = {};
    batch.n_tokens = 1;
    batch.embd = embd;
    batch.pos = &pos;
    batch.n_seq_id = &n_seq_id;
    batch.seq_id = seq_id_ptrs;

    llama_batch_allocr balloc(1);

    const bool ok = balloc.init(
        batch,
        vocab,
        &memory,
        /*n_embd*/ 1,
        /*n_seq_max*/ 4,
        /*output_all*/ false,
        /*allow_non_contiguous_pos*/ true);

    return ok ? 0 : 1;
}
