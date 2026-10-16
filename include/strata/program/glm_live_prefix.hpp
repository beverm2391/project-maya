#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::program {
// The one full-DSA branch still resident in the native model. This describes
// consumed tokens, not every emitted token: decode stops before feeding its
// final sample. Any request that starts writing clears the previous claim.
class GlmLivePrefix {
public:
    void clear() { tokens_.clear(); }
    void keep(const std::vector<int32_t>& prompt, const std::vector<int32_t>& emitted, int64_t position) {
        clear();
        if (position < 0 || (uint64_t) position > prompt.size() + emitted.size()) return;
        const size_t n = (size_t) position, input = std::min(n, prompt.size());
        tokens_.assign(prompt.begin(), prompt.begin() + input);
        if (n > input) tokens_.insert(tokens_.end(), emitted.begin(), emitted.begin() + (n-input));
    }
    size_t match(const std::vector<int32_t>& prompt, int64_t position) const {
        if (prompt.empty() || position <= 0) return 0;
        // The last prompt token must run to recreate logits after a rewind.
        const size_t limit = std::min({tokens_.size(), prompt.size()-1, (size_t) position});
        size_t n = 0;
        while (n < limit && tokens_[n] == prompt[n]) ++n;
        return n;
    }
private:
    std::vector<int32_t> tokens_;
};
} // namespace strata::program
