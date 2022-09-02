#include <iostream>
#include <vector>
#include <memory>

struct Base {
  virtual ~Base() = default;
};

struct Derived : public Base {
};

void test(const std::vector<std::unique_ptr<Base>>& base) {
}

int main() {
  std::vector<std::unique_ptr<Base>> vec;
  test(vec);
  return 0;
}
