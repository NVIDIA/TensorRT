/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_TOOLS_COMPANION_SO_H
#define TRT_TOOLS_COMPANION_SO_H

#include <filesystem>
#include <optional>
#include <string>

//! \file companionSo.h
//!
//! Locating the companion library that holds a safety engine's generated host code. Whether an engine
//! needs one is recorded in the engine, so a tool decides nothing here: it offers the library beside the
//! engine when there is one and lets the runtime reject a pair that does not match.

namespace toolsCompanionSo
{

//! \brief Resolve the companion library to load beside a safe engine.
//!
//! An explicit path is passed through, so a caller who names a library that is not there gets an error
//! from the runtime instead of silence. The default beside the engine is offered only when a file is
//! there, so an engine built without a companion library still loads. Only presence is asked: a file
//! that is there but unusable, truncated or not an ELF object, is left for dlopen to reject, since it
//! names the file and says what is wrong with it. An engine that needs a library and does not get one is
//! caught by the runtime, which reads the requirement out of the engine rather than from the caller.
//!
//! \param enginePath Path the engine was loaded from.
//! \param explicitPath Path given on the command line, or empty when none was.
//!
//! \return An absolute path to hand to createTRTGraph(), or std::nullopt when no companion library
//!         applies.
//!
//! \throws std::filesystem::filesystem_error when the filesystem cannot answer. The safe runtime rejects
//!         a relative path, so failing here names the path rather than leaving graph creation to
//!         complain about something further downstream.
[[nodiscard]] inline std::optional<std::string> resolveCompanionSoPath(
    std::string const& enginePath, std::string const& explicitPath = "")
{
    std::filesystem::path const candidate{explicitPath.empty() ? enginePath + ".so" : explicitPath};
    if (explicitPath.empty() && !std::filesystem::is_regular_file(candidate))
    {
        return std::nullopt;
    }
    return std::filesystem::absolute(candidate).lexically_normal().string();
}

} // namespace toolsCompanionSo

#endif // TRT_TOOLS_COMPANION_SO_H
