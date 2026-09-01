#include "imgui_platform.hpp"

#include <imgui.h>

#include <algorithm>
#include <limits>
#include <optional>

namespace
{

[[nodiscard]] constexpr auto
to_imgui_key(sf::Keyboard::Key key) -> ImGuiKey
{
  switch (key)
  {
  case sf::Keyboard::Key::Unknown   : return ImGuiKey_None;
  case sf::Keyboard::Key::A         : return ImGuiKey_A;
  case sf::Keyboard::Key::B         : return ImGuiKey_B;
  case sf::Keyboard::Key::C         : return ImGuiKey_C;
  case sf::Keyboard::Key::D         : return ImGuiKey_D;
  case sf::Keyboard::Key::E         : return ImGuiKey_E;
  case sf::Keyboard::Key::F         : return ImGuiKey_F;
  case sf::Keyboard::Key::G         : return ImGuiKey_G;
  case sf::Keyboard::Key::H         : return ImGuiKey_H;
  case sf::Keyboard::Key::I         : return ImGuiKey_I;
  case sf::Keyboard::Key::J         : return ImGuiKey_J;
  case sf::Keyboard::Key::K         : return ImGuiKey_K;
  case sf::Keyboard::Key::L         : return ImGuiKey_L;
  case sf::Keyboard::Key::M         : return ImGuiKey_M;
  case sf::Keyboard::Key::N         : return ImGuiKey_N;
  case sf::Keyboard::Key::O         : return ImGuiKey_O;
  case sf::Keyboard::Key::P         : return ImGuiKey_P;
  case sf::Keyboard::Key::Q         : return ImGuiKey_Q;
  case sf::Keyboard::Key::R         : return ImGuiKey_R;
  case sf::Keyboard::Key::S         : return ImGuiKey_S;
  case sf::Keyboard::Key::T         : return ImGuiKey_T;
  case sf::Keyboard::Key::U         : return ImGuiKey_U;
  case sf::Keyboard::Key::V         : return ImGuiKey_V;
  case sf::Keyboard::Key::W         : return ImGuiKey_W;
  case sf::Keyboard::Key::X         : return ImGuiKey_X;
  case sf::Keyboard::Key::Y         : return ImGuiKey_Y;
  case sf::Keyboard::Key::Z         : return ImGuiKey_Z;
  case sf::Keyboard::Key::Num0      : return ImGuiKey_0;
  case sf::Keyboard::Key::Num1      : return ImGuiKey_1;
  case sf::Keyboard::Key::Num2      : return ImGuiKey_2;
  case sf::Keyboard::Key::Num3      : return ImGuiKey_3;
  case sf::Keyboard::Key::Num4      : return ImGuiKey_4;
  case sf::Keyboard::Key::Num5      : return ImGuiKey_5;
  case sf::Keyboard::Key::Num6      : return ImGuiKey_6;
  case sf::Keyboard::Key::Num7      : return ImGuiKey_7;
  case sf::Keyboard::Key::Num8      : return ImGuiKey_8;
  case sf::Keyboard::Key::Num9      : return ImGuiKey_9;
  case sf::Keyboard::Key::Escape    : return ImGuiKey_Escape;
  case sf::Keyboard::Key::LControl  : return ImGuiKey_LeftCtrl;
  case sf::Keyboard::Key::LShift    : return ImGuiKey_LeftShift;
  case sf::Keyboard::Key::LAlt      : return ImGuiKey_LeftAlt;
  case sf::Keyboard::Key::LSystem   : return ImGuiKey_LeftSuper;
  case sf::Keyboard::Key::RControl  : return ImGuiKey_RightCtrl;
  case sf::Keyboard::Key::RShift    : return ImGuiKey_RightShift;
  case sf::Keyboard::Key::RAlt      : return ImGuiKey_RightAlt;
  case sf::Keyboard::Key::RSystem   : return ImGuiKey_RightSuper;
  case sf::Keyboard::Key::Menu      : return ImGuiKey_Menu;
  case sf::Keyboard::Key::LBracket  : return ImGuiKey_LeftBracket;
  case sf::Keyboard::Key::RBracket  : return ImGuiKey_RightBracket;
  case sf::Keyboard::Key::Semicolon : return ImGuiKey_Semicolon;
  case sf::Keyboard::Key::Comma     : return ImGuiKey_Comma;
  case sf::Keyboard::Key::Period    : return ImGuiKey_Period;
  case sf::Keyboard::Key::Apostrophe: return ImGuiKey_Apostrophe;
  case sf::Keyboard::Key::Slash     : return ImGuiKey_Slash;
  case sf::Keyboard::Key::Backslash : return ImGuiKey_Backslash;
  case sf::Keyboard::Key::Grave     : return ImGuiKey_GraveAccent;
  case sf::Keyboard::Key::Equal     : return ImGuiKey_Equal;
  case sf::Keyboard::Key::Hyphen    : return ImGuiKey_Minus;
  case sf::Keyboard::Key::Space     : return ImGuiKey_Space;
  case sf::Keyboard::Key::Enter     : return ImGuiKey_Enter;
  case sf::Keyboard::Key::Backspace : return ImGuiKey_Backspace;
  case sf::Keyboard::Key::Tab       : return ImGuiKey_Tab;
  case sf::Keyboard::Key::PageUp    : return ImGuiKey_PageUp;
  case sf::Keyboard::Key::PageDown  : return ImGuiKey_PageDown;
  case sf::Keyboard::Key::End       : return ImGuiKey_End;
  case sf::Keyboard::Key::Home      : return ImGuiKey_Home;
  case sf::Keyboard::Key::Insert    : return ImGuiKey_Insert;
  case sf::Keyboard::Key::Delete    : return ImGuiKey_Delete;
  case sf::Keyboard::Key::Add       : return ImGuiKey_KeypadAdd;
  case sf::Keyboard::Key::Subtract  : return ImGuiKey_KeypadSubtract;
  case sf::Keyboard::Key::Multiply  : return ImGuiKey_KeypadMultiply;
  case sf::Keyboard::Key::Divide    : return ImGuiKey_KeypadDivide;
  case sf::Keyboard::Key::Left      : return ImGuiKey_LeftArrow;
  case sf::Keyboard::Key::Right     : return ImGuiKey_RightArrow;
  case sf::Keyboard::Key::Up        : return ImGuiKey_UpArrow;
  case sf::Keyboard::Key::Down      : return ImGuiKey_DownArrow;
  case sf::Keyboard::Key::Numpad0   : return ImGuiKey_Keypad0;
  case sf::Keyboard::Key::Numpad1   : return ImGuiKey_Keypad1;
  case sf::Keyboard::Key::Numpad2   : return ImGuiKey_Keypad2;
  case sf::Keyboard::Key::Numpad3   : return ImGuiKey_Keypad3;
  case sf::Keyboard::Key::Numpad4   : return ImGuiKey_Keypad4;
  case sf::Keyboard::Key::Numpad5   : return ImGuiKey_Keypad5;
  case sf::Keyboard::Key::Numpad6   : return ImGuiKey_Keypad6;
  case sf::Keyboard::Key::Numpad7   : return ImGuiKey_Keypad7;
  case sf::Keyboard::Key::Numpad8   : return ImGuiKey_Keypad8;
  case sf::Keyboard::Key::Numpad9   : return ImGuiKey_Keypad9;
  case sf::Keyboard::Key::F1        : return ImGuiKey_F1;
  case sf::Keyboard::Key::F2        : return ImGuiKey_F2;
  case sf::Keyboard::Key::F3        : return ImGuiKey_F3;
  case sf::Keyboard::Key::F4        : return ImGuiKey_F4;
  case sf::Keyboard::Key::F5        : return ImGuiKey_F5;
  case sf::Keyboard::Key::F6        : return ImGuiKey_F6;
  case sf::Keyboard::Key::F7        : return ImGuiKey_F7;
  case sf::Keyboard::Key::F8        : return ImGuiKey_F8;
  case sf::Keyboard::Key::F9        : return ImGuiKey_F9;
  case sf::Keyboard::Key::F10       : return ImGuiKey_F10;
  case sf::Keyboard::Key::F11       : return ImGuiKey_F11;
  case sf::Keyboard::Key::F12       : return ImGuiKey_F12;
  case sf::Keyboard::Key::F13       : return ImGuiKey_F13;
  case sf::Keyboard::Key::F14       : return ImGuiKey_F14;
  case sf::Keyboard::Key::F15       : return ImGuiKey_F15;
  case sf::Keyboard::Key::Pause     : return ImGuiKey_Pause;
  }
  return ImGuiKey_None;
}

[[nodiscard]] constexpr auto
to_imgui_button(sf::Mouse::Button button) -> std::optional<std::int32_t>
{
  switch (button)
  {
  case sf::Mouse::Button::Left  : return 0;
  case sf::Mouse::Button::Right : return 1;
  case sf::Mouse::Button::Middle: return 2;
  case sf::Mouse::Button::Extra1: return 3;
  case sf::Mouse::Button::Extra2: return 4;
  }
  return std::nullopt;
}

template<typename KeyEvent>
constexpr void
add_key_event(const KeyEvent& event, bool down)
{
  ImGuiIO& io = ImGui::GetIO();
  io.AddKeyEvent(ImGuiMod_Ctrl, event.control);
  io.AddKeyEvent(ImGuiMod_Shift, event.shift);
  io.AddKeyEvent(ImGuiMod_Alt, event.alt);
  io.AddKeyEvent(ImGuiMod_Super, event.system);
  const auto key = to_imgui_key(event.code);
  if (key != ImGuiKey_None) { io.AddKeyEvent(key, down); }
}

} // namespace

namespace f1st
{

void
imgui_process_event(const sf::Event& event)
{
  ImGuiIO& io = ImGui::GetIO();
  if (event.is<sf::Event::FocusGained>()) { io.AddFocusEvent(true); }
  if (event.is<sf::Event::FocusLost>()) { io.AddFocusEvent(false); }
  if (const auto* text = event.getIf<sf::Event::TextEntered>())
  {
    if (text->unicode > 0U && text->unicode <= 0x10FFFFU)
    {
      io.AddInputCharacter(static_cast<std::uint32_t>(text->unicode));
    }
  }
  if (const auto* key = event.getIf<sf::Event::KeyPressed>())
  {
    add_key_event(*key, true);
  }
  if (const auto* key = event.getIf<sf::Event::KeyReleased>())
  {
    add_key_event(*key, false);
  }

  if (const auto* moved = event.getIf<sf::Event::MouseMoved>())
  {
    io.AddMousePosEvent( //
      static_cast<float>(moved->position.x),
      static_cast<float>(moved->position.y));
  }
  if (const auto* pressed = event.getIf<sf::Event::MouseButtonPressed>())
  {
    if (const auto button = to_imgui_button(pressed->button))
    {
      io.AddMouseButtonEvent(*button, true);
    }
  }
  if (const auto* released = event.getIf<sf::Event::MouseButtonReleased>())
  {
    if (const auto button = to_imgui_button(released->button))
    {
      io.AddMouseButtonEvent(*button, false);
    }
  }
  if (const auto* wheel = event.getIf<sf::Event::MouseWheelScrolled>())
  {
    const bool horizontal = wheel->wheel == sf::Mouse::Wheel::Horizontal;
    io.AddMouseWheelEvent(              //
      horizontal ? wheel->delta : 0.0F, //
      horizontal ? 0.0F : wheel->delta);
  }
}

void
imgui_platform_new_frame(const sf::WindowBase& window,
  std::chrono::steady_clock::time_point& previous_frame)
{
  ImGuiIO& io = ImGui::GetIO();
  const auto size = window.getSize();
  io.DisplaySize = {
    static_cast<float>(size.x),
    static_cast<float>(size.y),
  };

  const auto now = std::chrono::steady_clock::now();
  io.DeltaTime = previous_frame == std::chrono::steady_clock::time_point {}
    ? 1.0F / 60.0F
    : std::max(std::chrono::duration<float>(now - previous_frame).count(),
        std::numeric_limits<float>::min());
  previous_frame = now;

  const auto mouse = sf::Mouse::getPosition(window);
  io.AddMousePosEvent(           //
    static_cast<float>(mouse.x), //
    static_cast<float>(mouse.y));
}

} // namespace f1st
