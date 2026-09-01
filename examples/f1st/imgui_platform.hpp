#pragma once

#include <SFML/Window/Event.hpp>
#include <SFML/Window/WindowBase.hpp>

#include <chrono>

namespace f1st
{

void
imgui_process_event(const sf::Event& event);

void
imgui_platform_new_frame(const sf::WindowBase& window,
  std::chrono::steady_clock::time_point& previous_frame);

} // namespace f1st
