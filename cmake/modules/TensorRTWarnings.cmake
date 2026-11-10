# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)
include(CheckCXXCompilerFlag)

# Warnings that are treated as errors when the compiler supports them.
set(TRT_WARNINGS_AS_ERRORS
    all
    address-of-packed-member
    address
    ambiguous-reversed-operator
    bitwise-instead-of-logical
    dangling
    deprecated-copy-with-user-provided-copy
    deprecated-dynamic-exception-spec
    deprecated-implementations
    deprecated-redundant-constexpr-static-def
    deprecated-this-capture
    enum-compare
    header-hygiene
    implicit-fallthrough
    implicitly-unsigned-literal
    incompatible-function-pointer-types
    infinite-recursion
    misleading-indentation
    mismatched-tags
    missing-braces
    move
    null-conversion
    packed-non-pod
    parentheses
    pessimizing-move
    self-assign
    shift-sign-overflow
    string-concatenation
    string-conversion
    thread-safety
    undefined-var-template
    uninitialized-const-pointer
    uninitialized-const-reference
    uninitialized
    unreachable-code-fallthrough
    unused-but-set-variable
    unused-const-variable
    unused-exception-parameter
    unused-label
    unused-local-typedef
    unused-value
    unused-variable
    vexing-parse
    constant-logical-operand
    defaulted-function-deleted
    implicit-const-int-float-conversion
    inconsistent-missing-destructor-override
    inconsistent-missing-override
    nullability-completeness
    reorder-init-list
    tautological-compare
    unknown-warning-option
    unreachable-code-aggressive
    unused-private-field
    unreachable-code-break
    unreachable-code-continue
)

# Add -Werror=<warning> for each supported warning to C++ targets in the calling
# directory and its subdirectories. -Werror=<warning> also enables the warning.
function(trt_enable_warnings_as_errors)
    set(options)
    foreach(warning IN LISTS TRT_WARNINGS_AS_ERRORS)
        string(MAKE_C_IDENTIFIER "TRT_CXX_SUPPORTS_WERROR_${warning}" supported)
        check_cxx_compiler_flag("-Werror=${warning}" ${supported})
        if(${supported})
            list(APPEND options "-Werror=${warning}")
        endif()
    endforeach()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${options}>")
endfunction()
