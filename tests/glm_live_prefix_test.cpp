#include "strata/program/glm_live_prefix.hpp"
#include <cstdio>
#include <cstdlib>
using strata::program::GlmLivePrefix;
void check(bool ok) { if (!ok) std::abort(); }
int main() {
    GlmLivePrefix live;
    const std::vector<int32_t> prompt{1,2,3,4}, emitted{5,6,7};
    live.keep(prompt, emitted, 6); // final emitted token was not consumed
    check(live.match({1,2,3,4,5,6,7,8}, 6) == 6);
    check(live.match({1,2,3,4,5,6}, 6) == 5); // final prompt token recreates logits
    check(live.match({1,2,9,4}, 6) == 2); // a changed tool/template suffix
    check(live.match({9,2,3,4}, 6) == 0);
    check(live.match({1,2,3,4,5}, 2) == 2); // never beyond native position
    check(live.match({}, 6) == 0);
    live.clear(); // state changed without successful completion
    check(live.match(prompt, 4) == 0);
    live.keep(prompt, {}, 2); // cancellation during tokenwise prefill
    check(live.match(prompt, 2) == 2);
    live.keep(prompt, {}, 0); // batched cancellation reset the native state
    check(live.match(prompt, 0) == 0);
    live.keep(prompt, emitted, 8); // impossible state must make no cache claim
    check(live.match(prompt, 8) == 0);
    std::puts("glm_live_prefix_test: PASS");
}
