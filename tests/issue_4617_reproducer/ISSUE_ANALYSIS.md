# Issue #4617: Technical Analysis and Proposed Fix

## Problem Statement

When TensorRT is used from a dynamically loaded library (loaded via `dlopen()`), the program crashes during cleanup when the library is unloaded with `dlclose()`. This is caused by TensorRT calling `dlclose()` on `libnvinfer_builder_resource.so.10.x` from a static C++ object destructor.

## Root Cause Analysis

### The Problem with Static Destructors

In C++, static objects are destroyed in reverse order of their construction. When a shared library is unloaded with `dlclose()`, the following sequence occurs:

1. `dlclose()` is called on the user's library
2. C++ static destructors in the user's library are executed
3. If TensorRT has static objects that call `dlclose()` on other libraries
4. This can cause a crash due to:
   - Double-free issues
   - Use-after-free if other code still references the unloaded library
   - Undefined behavior due to destructor ordering issues in glibc

### Why This Happens

The glibc dynamic linker has specific rules about destructor execution order:
- Static C++ destructors are called during library unload
- The order is not guaranteed across different shared libraries
- If library A depends on library B, and both have static destructors, the order can be problematic

### The Specific Issue

TensorRT appears to have code similar to:

```cpp
// PROBLEMATIC CODE (hypothetical)
class BuilderResourceManager {
    void* handle;
public:
    BuilderResourceManager() {
        handle = dlopen("libnvinfer_builder_resource.so.10.x", RTLD_LAZY);
    }
    ~BuilderResourceManager() {
        if (handle) {
            dlclose(handle);  // ← This is the problem!
        }
    }
};

// Static instance - destructor called during library unload
static BuilderResourceManager gBuilderResource;
```

When the user's library is unloaded, this static destructor runs and calls `dlclose()`, which can cause crashes.

## Proposed Solutions

### Solution 1: Move dlclose to IBuilder Destructor (Recommended)

Instead of using a static object, manage the builder resource lifetime explicitly:

```cpp
class IBuilder {
public:
    virtual ~IBuilder() noexcept {
        // Unload builder resource when the last builder is destroyed
        BuilderResourceManager::release();
    }
};

class BuilderResourceManager {
    static void* handle;
    static std::atomic<int> refCount;
    static std::mutex mutex;
    
public:
    static void acquire() {
        std::lock_guard<std::mutex> lock(mutex);
        if (refCount++ == 0) {
            handle = dlopen("libnvinfer_builder_resource.so.10.x", RTLD_LAZY);
        }
    }
    
    static void release() {
        std::lock_guard<std::mutex> lock(mutex);
        if (--refCount == 0 && handle) {
            dlclose(handle);
            handle = nullptr;
        }
    }
};
```

**Advantages:**
- Explicit lifetime management
- No static destructor issues
- Works correctly with dlopen/dlclose
- Thread-safe with reference counting

### Solution 2: Use __attribute__((destructor))

Use a destructor function instead of a C++ static destructor:

```cpp
static void* gBuilderResourceHandle = nullptr;

__attribute__((constructor))
static void initBuilderResource() {
    gBuilderResourceHandle = dlopen("libnvinfer_builder_resource.so.10.x", RTLD_LAZY);
}

__attribute__((destructor))
static void cleanupBuilderResource() {
    if (gBuilderResourceHandle) {
        dlclose(gBuilderResourceHandle);
        gBuilderResourceHandle = nullptr;
    }
}
```

**Advantages:**
- Simpler than Solution 1
- `__attribute__((destructor))` functions are called at a different phase than C++ destructors
- Better control over cleanup order

**Disadvantages:**
- GCC/Clang specific (not portable to MSVC)
- Still has some ordering issues, though less severe

### Solution 3: Use RTLD_NODELETE Flag

When loading the builder resource, use `RTLD_NODELETE`:

```cpp
handle = dlopen("libnvinfer_builder_resource.so.10.x", RTLD_LAZY | RTLD_NODELETE);
```

**Advantages:**
- Simplest fix
- Prevents the library from being unloaded
- No destructor issues

**Disadvantages:**
- Library stays in memory until process exit
- May not be desirable for long-running processes

### Solution 4: Leak the Handle (Workaround)

Simply don't call `dlclose()` at all:

```cpp
class BuilderResourceManager {
    void* handle;
public:
    BuilderResourceManager() {
        handle = dlopen("libnvinfer_builder_resource.so.10.x", RTLD_LAZY);
    }
    ~BuilderResourceManager() {
        // Don't call dlclose - let the OS clean up at process exit
        // handle = nullptr;
    }
};
```

**Advantages:**
- Simplest to implement
- No crash issues

**Disadvantages:**
- Resource leak (though OS cleans up at process exit)
- Not a proper fix

## Recommended Implementation

**Primary Recommendation: Solution 1 (IBuilder Destructor)**

This is the cleanest and most robust solution. It provides:
1. Explicit lifetime management
2. Thread safety
3. Proper cleanup
4. No static destructor issues
5. Works correctly with dlopen/dlclose

**Alternative Recommendation: Solution 3 (RTLD_NODELETE)**

If Solution 1 is too complex to implement, using `RTLD_NODELETE` is a simple and effective workaround.

## Testing Strategy

1. **Basic Test**: Load and unload a library that uses TensorRT
2. **Stress Test**: Repeatedly load/unload the library
3. **Multi-threaded Test**: Load/unload from multiple threads
4. **Valgrind Test**: Check for memory leaks and use-after-free
5. **AddressSanitizer Test**: Detect memory errors

## Implementation Checklist

- [ ] Identify all locations where `libnvinfer_builder_resource.so` is loaded
- [ ] Remove static object that calls `dlclose()` in destructor
- [ ] Implement reference-counted resource manager
- [ ] Add resource acquisition to IBuilder constructor
- [ ] Add resource release to IBuilder destructor
- [ ] Add thread safety (mutex/atomic)
- [ ] Test with reproducer
- [ ] Test with AddressSanitizer
- [ ] Test with Valgrind
- [ ] Update documentation

## References

- [dlopen man page](https://man7.org/linux/man-pages/man3/dlopen.3.html)
- [GCC destructor attribute](https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html)
- [C++ Static Initialization Order Fiasco](https://en.cppreference.com/w/cpp/language/siof)
