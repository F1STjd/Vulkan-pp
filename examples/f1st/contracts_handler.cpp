#if !defined(__GNUC__) || defined(__clang__)

#else
// GCC provides a weak default in libstdc++exp, but PE-COFF (MinGW) cannot
// resolve it — see libstdc++ PR124263. A strong definition is required.
#include <contracts>
#include <cstdlib>

void
handle_contract_violation(const std::contracts::contract_violation& violation)
{
  std::contracts::invoke_default_contract_violation_handler(violation);
  if (violation.is_terminating()) { std::abort(); }
}

#endif