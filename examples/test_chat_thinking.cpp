#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include "utils.h"

int main() {
    for (const std::string prefix : {"<think></think", "<think>\n</think", "<think></think>"}) {
        assert(has_prefilled_thinking_end(prefix));
    }
    assert(!has_prefilled_thinking_end("<think"));
    assert(!has_prefilled_thinking_end("<think>reasoning"));
    assert(!has_prefilled_thinking_end("</thinking>"));
    assert(has_prefilled_thinking_end("<think>quoted </thinking> text</think>"));

    // Every byte split includes merged </ and ></ tokens and prefilled </think.
    const std::string text = "<think>some reasoning</think>";
    const auto closing = text.find("</think>");
    for (size_t split = closing; split < text.size(); ++split) {
        std::string suffix = text.substr(0, split);
        assert(advance_thinking_end(suffix, text.substr(split)));
    }
    std::string suffix = "<think";
    assert(!advance_thinking_end(suffix, ">reasoning "));
    assert(!advance_thinking_end(suffix, "</"));
    assert(!advance_thinking_end(suffix, "think"));
    assert(advance_thinking_end(suffix, ">"));
    suffix.clear();
    assert(!advance_thinking_end(suffix, "ordinary text </thinking>"));
    assert(suffix.size() <= 7);
    std::cout << "PASS: prefilled and token-split thinking boundaries\n";
}
