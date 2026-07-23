// Standalone translation unit for the tinyobjloader implementation.
//
// The implementation pulls in <windows.h> (long-path file I/O), which
// transitively declares SSE intrinsics with C linkage. Compiling that in a
// module unit that also does `import std;` makes GCC reject it with
// "conflicting language linkage". Keeping the implementation in this plain
// (non-module) TU avoids the clash; load.cppm only consumes the declarations.

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
