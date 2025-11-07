# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This file handles logic relating to building "fat" static libraries, that is, ones which contain other static libraries.
# Idiomatically, CMake would prefer we ship all static libraries as part of our release and ship a file which specifies the linkages.
# However, for various reasons, this is both inadvisable and annoying. Instead, we would prefer to ship one static library containing all dependencies.
#
# To do that, we need rules to bundle static libraries into other static libraries. Hence, this class.

define_property(TARGET
    PROPERTY BUNDLED_LIBRARY_TEMPLATE_PATH
    BRIEF_DOCS "File path to the template script which will be written to when calling target_bundle_libraries."
)

define_property(TARGET
    PROPERTY BUNDLE_LIBRARIES
    BRIEF_DOCS "The list of libraries that have been bundled into this target by target_bundle_libraries."
)

define_property(TARGET
    PROPERTY BUNDLE_LIBRARY_KNOWN_TYPE
    BRIEF_DOCS "Fallback type used when a target's TYPE is UNKNOWN_LIBRARY."
)

# Internal helper to prefix all messages with "[target_bundle_libraries]: ".
#
# \param mode The message mode to be passed to message(...)
# \param argn The message contents.
macro(__bundleMessage mode)
    message(${mode} "[target_bundle_libraries]: " ${ARGN})
endmacro()

# Illegal genex magic™
# Given a string containing a generator expression (var), edits the variable in-place to escape any generator expression literals.
# The escaped generator expression is then suitable for use within a LIST:TRANSFORM replacement block.
# This more or less allows for mapping lists to generator expressions, which can be recursively evaluated.
function(escape_generator_expression var)
    string(REPLACE ">" "__ANGLE_R__" ${var} "${${var}}")
    string(REPLACE "$" "$<1:$>" ${var} "${${var}}")
    string(REPLACE "," "$<COMMA>" ${var} "${${var}}")
    string(REPLACE "__ANGLE_R__" "$<ANGLE-R>" ${var} "${${var}}")
    return(PROPAGATE ${var})
endfunction()

# Recursively unwraps alias targets until finding the real target. Non-targets are returned verbatim.
# We need to ensure that the generated .mri script contains a deduplicated list of bundled targets
# so we need to resolve aliases, otherwise we may end up with multiple entries for the same target.
#
# \param target_name The name of the target to unwrap.
# \param result_var The variable to store the unwrapped target name in.
function(unwrapAlias target_name result_var)
    # First, try to unwrap common generator expressions that may wrap the target name.
    string(REGEX MATCH "\\$<LINK_LIBRARY:WHOLE_ARCHIVE,([a-zA-Z0-9_.:]+)>" _ ${target_name})
    if(TARGET ${CMAKE_MATCH_1})
        set(target_name ${CMAKE_MATCH_1})
    endif()

    string(REGEX MATCH "\\$<LINK_ONLY:([a-zA-Z0-9_.:]+)>" _ ${target_name})
    if(TARGET ${CMAKE_MATCH_1})
        set(target_name ${CMAKE_MATCH_1})
    endif()

    if(TARGET ${target_name})
        get_target_property(aliased_target ${target_name} ALIASED_TARGET)
        if(aliased_target)
            # Recursively unwrap in case there are multiple levels
            unwrapAlias(${aliased_target} unwrapped)
            set(${result_var} ${unwrapped} PARENT_SCOPE)
        else()
            # Not an alias, return the original name
            set(${result_var} ${target_name} PARENT_SCOPE)
        endif()
    else()
        # Not a target at all, return the original name
        set(${result_var} ${target_name} PARENT_SCOPE)
    endif()
endfunction()

# Internal function to retrieve the type of library for a given target.
# This function will fallback to the value of BUNDLE_LIBRARY_KNOWN_TYPE if it encounters an UNKNOWN_LIBRARY.
#
# \param lib    A target to evaluate the type for.
# \param outVar The output variable to store the type name in.
function(__get_lib_type lib outVar)
    get_target_property(libType ${lib} TYPE)
    if (${libType} STREQUAL UNKNOWN_LIBRARY)
        get_target_property(knownType ${lib} BUNDLE_LIBRARY_KNOWN_TYPE)
        if (NOT ${knownType} STREQUAL "knownType-NOTFOUND")
            set(libType ${knownType})
        endif()
        __bundleMessage(DEBUG "Using known type of unknown library ${lib}: ${knownType}")
    endif()
    set(${outVar} ${libType} PARENT_SCOPE)
endfunction()

# This is an internal-only function called the first time that target_bundle_libraries is called.
# It "registers" a target for bundling by creating the base template file and populating the target property BUNDLED_LIBRARY_TEMPLATE_PATH.
# Additionally, it registers the file generation logic for making the "final" script, as well as the custom command for running it after the build.
#
# \param lib          The mainLib from target_bundle_libraries.
# \param templatePath The file path to the template file that will be created.
function(__registerTargetForBundling lib templatePath)
    if(MSVC)
        set(scriptPath $<TARGET_FILE_DIR:${lib}>/archive-${lib}.bat)

        set(template "/OUT:\"$<TARGET_FILE:${lib}>\" \"$<TARGET_FILE:${lib}>\"\n")

        # Windows-syntax version of the same logic from the linux build below.
        # Main differences is that windows does not have `addlib`, and uses `\n` instead of `\n`. Additionally, the values need to be quoted to account for spaces in file paths.
        set(replaceExpr "\"$<IF:$<TARGET_EXISTS:\\1>,$<TARGET_FILE:\\1>,\\1>\"")
        escape_generator_expression(replaceExpr)
        string(APPEND template "$<TARGET_GENEX_EVAL:${lib},$<JOIN:$<LIST:TRANSFORM,$<TARGET_PROPERTY:${lib},BUNDLE_LIBRARIES>,REPLACE,(.+),${replaceExpr}>,\n>>\n")

        file(WRITE ${templatePath} ${template})
        file(GENERATE
            OUTPUT ${scriptPath}
            INPUT ${templatePath}
        )
        add_custom_command(TARGET ${lib} POST_BUILD
            COMMAND ${CMAKE_AR} /NOLOGO @\"${scriptPath}\"
            COMMAND ${CMAKE_COMMAND} -E echo "Bundled $<LIST:LENGTH,$<TARGET_PROPERTY:${lib},BUNDLE_LIBRARIES>> static libraries into target ${lib}. Script: ${scriptPath}"
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${lib}>
        )
    else()
        set(scriptPath $<TARGET_FILE_DIR:${lib}>/archive-${lib}.mri)

        set(template "create $<TARGET_FILE:${lib}>\n")
        string(APPEND template "addlib $<TARGET_FILE:${lib}>\n")
        # Expand BUNDLE_LIBRARIES into the appropriate chain of addlib commands needed.
        # BUNDLE_LIBRARIES will contain either (a) target names or (b) absolute file paths to libraries to include.
        # This first part will disambiguate between (a) and (b) by evaluating (a) to `addlib $<TARGET_FILE:lib>` and (b) to `addlib [[filepath]]`
        set(replaceExpr "addlib $<IF:$<TARGET_EXISTS:\\1>,$<TARGET_FILE:\\1>,\\1>")
        escape_generator_expression(replaceExpr)

        # The second part maps every element in BUNDLE_LIBRARIES to `replaceExpr` and evaluates the resulting replacement, which produces the final .mri file.
        string(APPEND template "$<TARGET_GENEX_EVAL:${lib},$<JOIN:$<LIST:TRANSFORM,$<TARGET_PROPERTY:${lib},BUNDLE_LIBRARIES>,REPLACE,(.+),${replaceExpr}>,\n>>\n")
        string(APPEND template "save\n")
        string(APPEND template "end\n")

        file(WRITE ${templatePath} ${template})
        file(GENERATE
            OUTPUT ${scriptPath}
            INPUT ${templatePath}
        )

        add_custom_command(TARGET ${lib} POST_BUILD
            COMMAND ${CMAKE_AR} -M < ${scriptPath}
            COMMAND ${CMAKE_RANLIB} $<TARGET_FILE:${lib}>
            COMMAND ${CMAKE_COMMAND} -E echo "Bundled $<LIST:LENGTH,$<TARGET_PROPERTY:${lib},BUNDLE_LIBRARIES>> static libraries into target ${lib}. Script: ${scriptPath}"
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${lib}>
        )
    endif()

    set_target_properties(${lib}
        PROPERTIES BUNDLED_LIBRARY_TEMPLATE_PATH ${templatePath}
    )
endfunction()

# Subcomponent of target_bundle_libraries which is responsible for walking the provided depLibs and recursively calling target_bundle_libraries.
# This macro must only be used within target_bundle_libraries.
#
# \param mainLib The current main library from target_bundle_libraries
# \param linkVis The link visibility from target_bundle_libraries
# \param argn    The dependencies to walk. Usually the INTERFACE_LINK_LIBRARIES of a target currently being bundled.
function(__bundleRecursiveDeps mainLib linkVis)
    if(ARGN)
        get_target_property(bundledLibs ${mainLib} BUNDLE_LIBRARIES)

        foreach(dep IN LISTS ARGN)
            unwrapAlias(${dep} dep)
            if(${dep} IN_LIST bundledLibs)
                continue() # Skip bundling of already-bundled libs to avoid many invocations of the same warnings.
            endif()

            if(TARGET ${dep})
                __get_lib_type(${dep} depType)
                if (${depType} STREQUAL STATIC_LIBRARY)
                    target_bundle_libraries(${mainLib} ${linkVis} ${dep})
                elseif(${depType} STREQUAL INTERFACE_LIBRARY)
                    # For interface libraries, we want to add all of the static libraries they may be pointing to, without the library itself (since it is not a static).
                    get_target_property(interfaceLibs ${dep} INTERFACE_LINK_LIBRARIES)
                    __bundleRecursiveDeps(${mainLib} ${linkVis} ${interfaceLibs})
                elseif(${depType} STREQUAL SHARED_LIBRARY)
                    # Skip SO's, since static libraries at the SO-boundary should not be bundled into the target mainLib.
                else()
                    __bundleMessage(DEBUG "Skipping unhandled dependency ${dep} (a dependency of ${bundledLib}) with type ${depType} in ${mainLib}")
                endif()
            else()
                __bundleMessage(DEBUG "Failed to recursively bundle dependency ${dep} (a dependency of ${bundledLib}) in ${mainLib}")
            endif()
        endforEach()
    endif()
endfunction()

# This function acts as a replacement target_link_libraries, instead linking
# one or more "bundled" static libraries into a "main" static library.
# The bundled libs are embedded inside the main lib during the post-build step.
# CMake interface properties for definitions, etc. are propagated using the specified visibility.
#
# This function is recursive. Any static dependencies of any bundled library will also be bundled into the mainLib.
#
# \param mainLib The main library that other libraries are to be bundled into.
# \param linkVis The link visiblity for interface properties. One of PRIVATE or PUBLIC.
#                Using PRIVATE hides all interface properties of bundled libraries from consumers of this library.
#                Using PUBLIC will share interface properties with dependents.
# \param argn    Variadic arguments - The list of targets to be linked into the mainLib.
function(target_bundle_libraries mainLib linkVis)
    if (NOT ${linkVis} STREQUAL PUBLIC AND NOT ${linkVis} STREQUAL PRIVATE)
        __bundleMessage(FATAL_ERROR "Error: Called target_bundle_libraries with unknown visibility ${linkVis}. Must be either \"PUBLIC\" or \"PRIVATE\".")
    endif()

    # TODO: Allow direct insertion of absolute file paths when CMAKE_LINK_LIBRARIES_ONLY_TARGETS is off, instead of always forcing targets.
    if (NOT TARGET ${mainLib})
        __bundleMessage(FATAL_ERROR "Error: Called target_bundle_libraries for target ${mainLib}, but no target with that name is known.")
    endif()

    __get_lib_type(${mainLib} mainLibType)
    if (NOT mainLibType STREQUAL STATIC_LIBRARY)
        __bundleMessage(FATAL_ERROR "Error: Called target_bundle_libraries for target ${mainLib}, but it is not a static library.")
    endif()

    get_target_property(isMainLibImported ${mainLib} IMPORTED)
    if(isMainLibImported)
        __bundleMessage(FATAL_ERROR "Error: Called target_bundle_libraries for target ${mainLib}, but it is an imported target.")
    endif()

    if(NOT ARGN)
        __bundleMessage(WARNING "Called target_bundle_libraries with no libraries for target ${mainLib}")
    endif()

    __bundleMessage(DEBUG "Bundling ${ARGN} into ${mainLib}")

    get_target_property(templatePath ${mainLib} BUNDLED_LIBRARY_TEMPLATE_PATH)
    if(NOT EXISTS ${templatePath})
        if(MSVC)
            set(templatePath ${PROJECT_BINARY_DIR}/archive-${mainLib}.bat.template)
        else()
            set(templatePath ${PROJECT_BINARY_DIR}/archive-${mainLib}.mri.template)
        endif()
        __bundleMessage(STATUS "Registering default script path ${templatePath} for target ${mainLib}")
        __registerTargetForBundling(${mainLib} ${templatePath})
    endif()

    get_target_property(bundledLibs ${mainLib} BUNDLE_LIBRARIES)
    if (NOT bundledLibs)
        set(bundledLibs "")
    endif()

    foreach(bundledLib IN LISTS ARGN)
        unwrapAlias(${bundledLib} bundledLib)
        __get_lib_type(${bundledLib} bundledLibType)

        if(${bundledLibType} STREQUAL INTERFACE_LIBRARY)
            # Interface libraries are not bundled, since they do not contain any static libraries.
            # Their dependencies will be bundled recursively in the next step.
            continue()
        endif()

        if (NOT ${bundledLibType} STREQUAL STATIC_LIBRARY AND NOT ${bundledLibType} STREQUAL OBJECT_LIBRARY)
            __bundleMessage(FATAL_ERROR "Attempted to bundle ${bundledLibType} library ${bundledLib} into target ${mainLib} (only static and object libraries may be bundled)")
        endif()

        if (${bundledLibType} STREQUAL STATIC_LIBRARY)
            list(APPEND bundledLibs ${bundledLib})
        else()
            # Exclude object libs from the BUNDLE_LIBRARIES property as they get added into the static lib using normal means.
        endif()
    endforeach()

    list(REMOVE_DUPLICATES bundledLibs)
    set_target_properties(${mainLib}
        PROPERTIES BUNDLE_LIBRARIES "${bundledLibs}"
    )

    foreach(bundledLib IN LISTS ARGN)
        unwrapAlias(${bundledLib} bundledLib)
        __get_lib_type(${bundledLib} bundledLibType)

        # Recursively bundle all static dependencies of each lib to be bundled.
        get_target_property(depLibs ${bundledLib} LINK_LIBRARIES)
        __bundleRecursiveDeps(${mainLib} ${linkVis} ${depLibs})

        # Since we want the main library to be linkable standalone, we need both the LINK_LIBRARIES and INTERFACE_LINK_LIBRARIES bundled in.
        # Otherwise, private static dependencies may be lost.
        get_target_property(depLibs ${bundledLib} INTERFACE_LINK_LIBRARIES)
        __bundleRecursiveDeps(${mainLib} ${linkVis} ${depLibs})

        # Use `target_link_libraries` to propagate INTERFACE definitions from bundled libs
        # BUILD_LOCAL_INTERFACE prevents clients from seeing this internal link relationship
        # COMPILE_ONLY prevents the bundled lib from appearing in the link command redundantly
        if (${bundledLibType} STREQUAL STATIC_LIBRARY OR ${bundledLibType} STREQUAL INTERFACE_LIBRARY)
            target_link_libraries(${mainLib} ${linkVis}
                $<BUILD_LOCAL_INTERFACE:$<COMPILE_ONLY:${bundledLib}>>
            )
        else()
            # Include Object Libraries as full libraries, since they do not get added by the bundling stage.
            # To do this without breaking the link dependency logic, we need to steal the target objects and link the lib as local + compile only.
            target_link_libraries(${mainLib} ${linkVis}
                $<BUILD_LOCAL_INTERFACE:$<COMPILE_ONLY:${bundledLib}>>
            )
            target_sources(${mainLib} PRIVATE $<TARGET_OBJECTS:${bundledLib}>)
        endif()

        # require that bundled libs are built before we try to bundle them
        add_dependencies(${mainLib} ${bundledLib})
    endforeach()

    if(NOT EXISTS ${templatePath})
        __bundleMessage(FATAL_ERROR "Template file ${templatePath} for target ${mainLib} does not exist.")
    endif()
endfunction()
