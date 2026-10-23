export module vkpp.cstring_view;

import std;

namespace vkpp
{

inline
void
cstring_view_error([[maybe_unused]] std::string_view reason)
{}

export class cstring_view
{
public:
  template<std::size_t Size>
  consteval cstring_view(const char (&literal)[ Size ])
  : view_ {literal, Size - 1UZ}
  {
    if (literal[ Size - 1UZ ] != '\0' || view_.contains('\0'))
    {
      cstring_view_error("a cstring_view literal needs to be null-terminated");
    }
  }

  consteval cstring_view(std::string_view literal_view) : view_ {literal_view}
  {
    if (literal_view.data()[ literal_view.size() ] != '\0')
    {
      cstring_view_error(
        "a cstring_view literal needs a view that is null-terminated");
    }
  }

  constexpr cstring_view(const std::string& owner) noexcept : view_ {owner} {}

  cstring_view(std::string&&) = delete;

  [[nodiscard]]
  constexpr
  auto
  c_str() const noexcept -> const char*
  { return view_.data(); }

  [[nodiscard]]
  constexpr
  auto
  data() const noexcept -> const char*
  { return view_.data(); }

  [[nodiscard]]
  constexpr
  auto
  size() const noexcept -> std::size_t
  { return view_.size(); }

  [[nodiscard]]
  constexpr
  auto
  empty() const noexcept -> bool
  { return view_.empty(); }

  constexpr
  void
  remove_prefix(std::size_t count) noexcept
  { view_.remove_prefix(count); }

  constexpr
  operator std::string_view() const noexcept
  { return view_; }

  friend constexpr
  auto
  operator ==(const cstring_view&, const cstring_view&) -> bool = default;

private:
  std::string_view view_ {""};
};

} // namespace vkpp
