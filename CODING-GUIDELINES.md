
## TensorRT C++ Coding Guidelines

The TensorRT C++ Coding Guidelines are derived from several sources, primarily:

- [AUTOSAR C++ 2014](https://www.autosar.org/fileadmin/user_upload/standards/adaptive/17-03/AUTOSAR_RS_CPP14Guidelines.pdf)
- [MISRA C++ 2008](https://www.misra.org.uk/Activities/MISRAC/tabid/171/Default.aspx)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

------

#### Namespaces

1. *MISRA C++: 2008 Rule 7-3-1*
   Global namespace shall only contain main, namespace declarations and extern "C" declarations. Use explicit or anon namespaces for everything else.
2. Closing braces of namespaces should have a comment saying the namespace it closes:
```cpp
namespace foo
{
...
} // namespace foo
```

#### Constants
1. Prefer `const` or `constexpr` variables over `#defines` whenever possible, as the latter are not visible to the compiler.
2. *MISRA C++: 2008 Rule 7-1-1 and 7-1-2*
   A variable that is not modified after its initialization should be declared as `const`.
3. For naming of constants, see the Naming section of this document.



#### Literals
1. Except `0`  (only used in comparison for checking signness/existence/emptiness) and `nullptr`, `true`, `false`, all other literals should only be used for variable initialization.
   Example:
```cpp
if (nbInputs == 2U){/*...*/}
```
   Should be changed to:
```cpp
constexpr size_t kNbInputsWBias = 2U;
if (nbInputs == kNbInputsWBias) {/*...*/}
```


#### Brace Notation
1. Use the [Allman indentation](https://en.wikipedia.org/wiki/Indent_style#Allman_style) style.
2. Put the semicolon for an empty `for` or `while` loop in a new line.
3. *AUTOSAR C++14 Rule 6.6.3*, *MISRA C++: 2008 6-3-1*
   The statement forming the body of a `switch`, `while`, `do .. while` or `for` statement shall be a compound statement. (use brace-delimited statements)
4. *AUTOSAR C++14 Rule 6.6.4*, *MISRA C++: 2008 Rule 6-4-1*
   `If` and `else` should always be followed by brace-delimited statements, even if empty or a single statement.


#### Naming
1. Filenames
   * Camel case with first letter lowercase: `thisIsASubDir` and `thisIsAFilename.cpp`
   * *NOTE*: All files involved in the compilation of a compilation target (.exe/.so) must have filenames that are case-insensitive unique.

2. Types
   * All types (including, but not limited to, class names) are [camel case](https://en.wikipedia.org/wiki/Camel_case) with uppercase first letter. Example: `FooBarClass`

3. Local variables, methods and namespaces
   * Camel case with first letter lowercase. Example: `localFooBar`

4. Non-magic-number global variables that are non-static and not defined in anonymous namespace
   * Camel case prefixed by a lower case 'g'. Example: `gDontUseGlobalFoos`

5. Non-magic-number global variables that are static or defined in an anonymous namespace
   * Camel case prefixed by a lower case  's'. Example: `sMutableStaticGlobal`

6. Locally visible static variable
   * Camel case with lowercase prefix ''s" as the first letter of the name. Example: `static std::once_flag sCaskInitOnce;`

7. Public, private and protected class member variables
   * Camelcase prefixed with an 'm': `mNbFooValues`.
   * Public member variables do not require the 'm' prefix but it is highly encouraged to use the prefix when needed to improve code clarity, especially in cases where the class is a base class in an inheritance chain.

8. Constants
   * Enumerations, global constants, static constants at class-scope and function-scope magic-number/literal constants are uppercase snakecase with prefix 'k':
```cpp
const int kDIGIT_NUM = 10;
```
> *NOTE*: Function-scope constants that are not magic numbers or literals are named like non-constant variables:
```cpp
const bool pass = a && b;
```

9. Macros
   * See [Constants](CODING-GUIDELINES.md#constants), which are preferred over `#define`.
   * If you must use macros, however, follow uppercase snakecase: `FOO_VERSION`

Notes:
* In general we don't use [hungarian notation](https://en.wikipedia.org/wiki/Hungarian_notation), except for 'apps hungarian' in some cases such as 'nb' in a variable name to indicate count: `mNbTensorDescriptors`
* If a constructor's parameter name `foo` conflicts with a public member name `foo`, add a trailing underscore to the parameter name: `foo_`.
* *MISRA C++: 2008 Rule 2-13-4*
  Literal suffixes should be upper case. For example, use `1234L` instead of `1234l`.


#### Tabs vs Spaces
1. Use only spaces. Do not use tabs.
2. Indent 4 spaces at a time. This is enforced automatically if you format your code using our clang-format config.


#### Formatting
1. Use the [LLVM clang-format](https://clang.llvm.org/docs/ClangFormat.html) tool for formatting your changes prior to submitting the PR.
2. Use a maximum of 120 characters per line. The auto formatting tool will wrap longer lines.
3. Exceptions to formatting violations must be justified on a per-case basis. Bypassing the formatting rules is discouraged, but can be achieved for exceptions as follows:
```cpp
// clang-format off
// .. Unformatted code ..
// clang-format on
```


#### Pointers and Memory Allocation
1. *AUTOSAR C++ 2014: 18-5-2/3*
   Use smart pointers for allocating objects on the heap.
2. When picking a smart pointer, prefer `unique_ptr` for single resource ownership and `shared_ptr` for shared resource ownership. Use `weak_ptr` only in exceptional cases.
3. Do not use smart pointers that have been deprecated in C++11.


#### Comments
1. C++ comments are required. C comments are not allowed except for special cases (inline).
2. C++ style for single-line comments. `// This is a single line comment`
3. In function calls where parameters are not obvious from inspection, it can be helpful to use an inline C comment to document the parameter for readers:
```cpp
doSomeOperation(/* checkForErrors = */ false);
```
4. If the comment is a full sentence, it should be capitalized i.e. start with capital letter and punctuated properly.
5. Follow [Doxygen rules](http://www.doxygen.nl/manual/docblocks.html) for documenting new class interfaces and function prototypes.
* For C++-style single-line comments use `//!`.
* For class members, use `//!<`.
```cpp
//! This is a Doxygen comment
//! in C++ style

struct Foo
{
    int x; //!< This is a Doxygen comment for members
}
```


#### Disabling Code
1. Use `#if` / `#endif` to disable code, preferably with a mnemonic condition like this:
```cpp
#if DEBUG_CONVOLUTION_INSTRUMENTATION
// ...code to be disabled...
#endif
```

```cpp
// Alternative: use a macro which evaluates to a noop in release code.
#if DEBUG_CONVOLUTION_INSTRUMENTATION
# define DEBUG_CONV_CODE(x) x
#else
# define DEBUG_CONV_CODE(x)
#endif
```

2. *MISRA C++: 2008 Rule 0-1-9*, *AutoSAR C++ 2014: 6-0-1*
   Dead code is forbidden in safety-critical software - you may not use compile-time expressions and DCE to disable code. However, this technique can be useful elsewhere (e.g. tools, tests) to help prevent bitrot.

```cpp
// Not allowed in safety-critical code.
const bool gDisabledFeature = false;

void foo()
{
   if (gDisabledFeature)
   {
       doSomething();
   }
}
```

3. *MISRA C++: 2008 Rule 2-7-2 and 2-7-3*
   Do NOT use comments to disable code. Use comments to explain code, not hide it.


#### Exceptions
1.  Exceptions must not be thrown across library boundaries.


#### Casts
1. Use the least forceful cast necessary, or no cast if possible, to help the compiler diagnose unintended consequences.
2. Casting a pointer to a `void*` should be implicit (except if removing `const`).
3. *MISRA C++: 2008 Rule 5-2-5*
   Casting should not remove any `const` or `volatile` qualification from the type of a pointer or reference.
4. *MISRA C++: 2008 Rule 5-2-4*
   Do not use C-style casts (other than void casts) and functional notation casts (other than explicit constructor calls).
6. Casting from a `void*` to a `T*` should be done with `static_cast`, not `reinterpret_cast`, since the latter is more forceful.
7. Use `reinterpret_cast` as a last resort, where `const_cast` and `static_cast` won't work.
8. Avoid `dynamic_cast`.


#### Expressions
1. *MISRA C++: 2008 Rule 6-2-1*
   Do not use assignment operator in subexpressions.
```cpp
// Not compliant
x = y = z;

// Not compliant
if (x = y)
{
    // ...
}
```


#### Ternary operator
1. *AUTOSAR C++ 2014: 7-1-1*
   Ternary operator should not be used as a sub-expression. Ternary operator expressions should be encapsulated with braces. Example:
```cpp
const auto var = (condition0 ? a : (condition1 ? b : c));
```
   should be changed to:
```cpp
const auto d = (condition1 ? b : c);
const auto var = (condition0 ? a : d);
```


#### Statements
1. When practical, a `switch` statement controlled by an `enum` should have a case for each enum value and not have a default clause so that we get a compile-time error if a new enum value is added.
2. *MISRA C++:2008 Rules 6-4-3, 6-4-4, and 6-4-5*
   Switch statements should be well structured.  An informal guideline is to treat switch statements as structured multi-way branches and not "glorified gotos" such as:
```cpp
// Not compliant
switch (x) case 4: if (y) case 5: return 0; else default: return 1;
```
3. The "well structured" requirement prohibits fall-though except from one case label to another.   Each case clause must be terminated in a break or throw.  If a case clause has multiple statements, the braces are optional.  The following example illustrates these requirements:
```cpp
switch (x)
{
case 0:         // Fall-through allowed from case 0: to case 1: since case 0 is empty.
case 1:
    a();
    b();
    break;
case 2:
case 4:
{              // With optional braces
    c();
    d();
    break;
}
case 5:
    c();
    throw 42;  // Terminating with throw is okay
default:
    throw 42;
}
```

4. *MISRA C++:2008 Rule 6-4-3*
   Ending a case clause with return is not allowed.
5. If a switch clause is a compound statement, put the break inside the braces.
```cpp
switch (x)
{
case 0:
case 1:
{
    y();
    z();
    break;
}
...other cases...
}
```

#### Functions
1. Avoid declaring large functions as `inline`, absent a quantifiable benefit.  Remember that functions defined in class declarations are implicitly inline.
2. Rather than using the `static` keyword to mark a function as having internal linkage, prefer to use anonymous namespaces instead.
3. *MISRA C++:2008 Rule 0-1-10*
   Every defined function must be called at least once. That is, do not have unused methods.
4. *MISRA C++:2008 Rule 8-4-2*
   Parameter names should be consistent across function definition and corresponding function declarations.


#### Forward declarations and extern variables

1. *MISRA C++: 2008 Rule 3-2-3*
   For safety critical code, a type, object or function that is used in multiple translation units shall be declared in one and only one file.
   * This means we cannot forward declare incomplete types in files where they are needed. Instead, we should put forward declarations in header files, and include these header files as needed.


#### Structures and Classes
1. *MISRA C++: 2008 Rule 14-7-1*
   All class templates, function templates, class template member functions and class template static members shall be instantiated at least once. This prevents use of uninitialized variables.
2. *MISRA C++: 2008 Rule 11-01*
   If class is not a *Plain Old Data Structure*, then its data members should be private.


#### Preprocessor Directives
1. *MISRA C++: 2008 Rule 16-0-2*
   `#define` and `#undef` of macros should be done only at global namespace.
2. Avoid the use of `#ifdef` and `#ifndef` directives (except in the case of header include guards). Prefer to use `#if defined(...)` or `#if !defined(...)` instead. The latter syntax is more consistent with C syntax, and allows you to use more complicated preprocessor conditionals, e.g.:
```cpp
#if defined(FOO) || defined(BAR)
void foo();
#endif // defined(FOO) || defined(BAR)
```

3. When nesting preprocessor directives, use indentation after the hash mark (#). For example:
```cpp
#if defined(FOO)
# if FOO == 0
#  define BAR 0
# elif FOO == 1
#  define BAR 5
# else
#  error "invalid FOO value"
# endif
#endif
```

4. Do not use `#pragma` once as include guard.
5. Use a preprocessor guard. It's standard-conforming and modern compilers are smart enough to open the file only once.
   * The guard name must have prefix `TRT_` followed by the filename, all in caps. For a header file named `FooBarHello.h`, name the symbol as `TRT_FOO_BAR_HELLO_H`.
   * Only use the file name to create the symbol. Unlike the Google C++ guideline, we do not include the directory names in the symbol. This is because we ensure all filenames are unique in the compilation unit.
   * Do not use prefix with underscore. Such symbols are reserved in C++ standard for compilers or implementation.
   * Do not use trailing underscore for the symbol. We differ in this from Google C++ guideline, which uses trailing underscore: `TRT_FOO_BAR_HELLO_H_`
```cpp
#ifndef TRT_FOO_BAR_HELLO_H
#define TRT_FOO_BAR_HELLO_H
// ...
#endif // TRT_FOO_BAR_HELLO_H
```

6. *AUTOSAR C++ 2014: 7-1-6*
   Use `using` instead of `typedef`.


#### Signed vs Unsigned Integers
1. Use signed integers instead of unsigned, except for  the cases below.
* The integer is a bitmap - use an unsigned type, since sign extension could lead to surprises.
* The integer is being used with an external library that expects an unsigned integer.  A common example is a loop that compares against `std::vector::size()`, such as:
```cpp
for (size_t i = 0; i < mTensors.size(); ++i) // preferred style
```
* Using only signed integers for the above would lead to prolixity and perhaps unsafe narrowing:
```cpp
for (int i = 0; i < static_cast<int>(mTensors.size()); ++i)
```


#### Special Considerations for API
1. The API consists, with very few exceptions, of methodless structs and pure virtual interface classes.
2. API class methods should be either virtual or inline.
3. The API does not use  integral types with platform-dependent sizes, other than `int`, `unsigned`, and `bool`.   `size_t` should be used only for sizes of memory buffers.
4. The API does not use any aggregate types (e.g. `std::string`) which may be compiled differently with different compilers and libraries.
5. The API minimizes dependencies on system headers - currently only `<cstddef>` and `<cstdint>`.
6. Memory ownership may not be transferred across API boundaries - any memory allocated inside a library must be freed inside the library.
7. The API should be C++03.
8. New methods should be added at the end of interfaces so as to preserve v-table compatibility (compilers don't guarantee this, but de facto it works.)
9. Avoid optional arguments to functions, since they can make it difficult to extend interfaces.
10. Do not throw exceptions across library boundaries.
11. Document all APIs with doxygen.


#### Common Pitfalls

1. C headers should not be used directly.
   - Example: Use `<cstdint>` instead of  `<stdint.h>`
2. Do not use C library functions, whenever possible.
   * Use brace initialization or `std::fill_n()` instead of `memset()`. This is especially important when dealing with non-[POD types](http://en.cppreference.com/w/cpp/concept/PODType). In the example below, using `memset()` will corrupt the vtable of `Foo:`
```cpp
struct Foo {
    virtual int getX() { return x; }
    int x;
};
...

// Bad: use memset() to initialize Foo
{
    Foo foo;
    memset(&foo, 0, sizeof(foo)); // Destroys hiddien virtual-function-table pointer!
}
// Good: use brace initialization to initialize Foo
{
    Foo foo = {};
}
```

2. When specifying pointers to `const` data, the pointer itself may be `const`, in some usecases.
```cpp
char const * const errStr = getErrorStr(status);
```

----

## Appendix

####  Abbreviation Words and Compound Words as Part of Names

* Abbreviation words, which are usually fully-capitalized in literature, are treated as normal words without special capitalization, e.g. `gpuAllocator`, where GPU is converted to `gpu` before constructing the camel case name.
* Compound words, which are usually used in full in literature, e.g. `runtime`, can be abbreviated into fully capitalized letters, e.g. `RT` in NvInferRT.h.

####  Terminology

* *CUDA code* is code that must be compiled with a CUDA compiler. Typically, it includes:
   * Declaration or definition of global or static variables with one of the following CUDA keywords: `__device__`, `__managed__` and `__constant__`.
   * Declaration or definition of device functions decorated with `__device__`.
   * Declaration or definition of kernels decorated with `__global__`.
   * Kernel launching with <<<...>>> syntax.

> NOTE:
   * Definition of kernel function pointer type aliases is not device code, e.g. `typedef __global__ void(*KernelFunc)(void* /*arg*/);`.
   * Definition of pointers to kernel functions is not device code, either, e.g. `__global__ void(*KernelFunc)(void* /*arg*/) = getKernelFunc(parameters);` .
   * Kernel launching with the CUDA runtime/driver API's, e.g. `cuLaunch` and `cudaLaunch`, is not CUDA code.

----

## NVIDIA Copyright

1. All TensorRT Open Source Software code should contain an NVIDIA copyright header that includes the current year.  The following block of text should be prepended to the top of all OSS files.  This includes .cpp, .h, .cu, .py, and any other source files which are compiled or interpreted.
```cpp
/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```
