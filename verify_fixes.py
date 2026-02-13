#!/usr/bin/env python3
"""
Verification script to ensure the fixes for Issue #4607 work correctly.
This script performs basic sanity checks on the modified files.
"""

import sys
import os

# Add Polygraphy to path
sys.path.insert(0, '/vercel/sandbox/tools/Polygraphy')

def verify_to_input_fix():
    """Verify that to_input.py has the correct fix."""
    print("Verifying to_input.py fix...")
    
    with open('/vercel/sandbox/tools/Polygraphy/polygraphy/tools/data/subtool/to_input.py', 'r') as f:
        content = f.read()
    
    # Check that the buggy line is NOT present
    if '[OrderedDict()] *' in content:
        print("  ❌ FAILED: Buggy list multiplication still present!")
        return False
    
    # Check that the fixed line IS present
    if '[OrderedDict() for _ in range(' in content:
        print("  ✓ PASSED: Fixed list comprehension is present")
        return True
    else:
        print("  ❌ FAILED: Fixed list comprehension not found!")
        return False


def verify_reduce_validation():
    """Verify that reduce.py has the validation logic."""
    print("\nVerifying reduce.py validation...")
    
    with open('/vercel/sandbox/tools/Polygraphy/polygraphy/tools/debug/subtool/reduce.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('args.reduce_inputs and not self.arg_groups[DataLoaderArgs].is_using_random_data()', 
         'Check for reduce_inputs and custom data loader'),
        ('num_iterations > 1', 
         'Check for multiple iterations'),
        ('only supports a single input iteration when input reduction is enabled', 
         'Error message about single iteration limitation'),
        ('--no-reduce-inputs', 
         'Workaround suggestion'),
    ]
    
    all_passed = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ PASSED: {description}")
        else:
            print(f"  ❌ FAILED: {description} not found!")
            all_passed = False
    
    return all_passed


def verify_documentation():
    """Verify that documentation has been updated."""
    print("\nVerifying documentation update...")
    
    with open('/vercel/sandbox/tools/Polygraphy/examples/cli/debug/02_reducing_failing_onnx_models/README.md', 'r') as f:
        content = f.read()
    
    checks = [
        ('only supports a single input iteration', 
         'Single iteration limitation mentioned'),
        ('--no-reduce-inputs', 
         'Workaround with --no-reduce-inputs'),
        ('fallback shape inference', 
         'Technical explanation'),
    ]
    
    all_passed = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ PASSED: {description}")
        else:
            print(f"  ❌ FAILED: {description} not found!")
            all_passed = False
    
    return all_passed


def verify_imports():
    """Verify that the modified files can be imported without errors."""
    print("\nVerifying imports...")
    
    try:
        # Try to import the modified modules
        from polygraphy.tools.data.subtool import to_input
        print("  ✓ PASSED: to_input module imports successfully")
        
        from polygraphy.tools.debug.subtool import reduce
        print("  ✓ PASSED: reduce module imports successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ FAILED: Import error: {e}")
        return False


def main():
    print("=" * 70)
    print("Verification Script for GitHub Issue #4607 Fixes")
    print("=" * 70)
    
    results = []
    
    # Run all verification checks
    results.append(("to_input.py fix", verify_to_input_fix()))
    results.append(("reduce.py validation", verify_reduce_validation()))
    results.append(("Documentation update", verify_documentation()))
    results.append(("Module imports", verify_imports()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ ALL VERIFICATIONS PASSED!")
        print("\nThe fixes for GitHub Issue #4607 have been successfully implemented:")
        print("  1. Fixed list padding bug in to_input.py")
        print("  2. Added validation in reduce.py for multiple iterations")
        print("  3. Updated documentation with limitation and workarounds")
        return 0
    else:
        print("\n❌ SOME VERIFICATIONS FAILED!")
        print("Please review the failed checks above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
