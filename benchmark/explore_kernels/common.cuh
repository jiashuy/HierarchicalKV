#pragma once
#include <iostream>

#define FETCH_FLOAT4(start) (reinterpret_cast<float4*>(&(start))[0])

void two_lines() {
  std::cout << "=================================================================================================\n";
}
void one_line() {
  std::cout << "-------------------------------------------------------------------------------------------------\n";
}