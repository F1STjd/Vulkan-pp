#pragma once

// Clang (and clangd) do not implement C++26 contracts yet. GCC needs
// -fcontracts.
#if defined(__clang__)
#define PRE(...)
#define POST(...)
#else
#define PRE(cond) pre(cond)
#define POST(cond) post(cond)
#endif
