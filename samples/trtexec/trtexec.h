/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRTEXEC_H
#define TRTEXEC_H

#include "sampleEngines.h"

namespace sample
{

//! \brief Reusable trtexec implementation, separated from main() so that
//! custom command-line tools can be built on top of trtexec's engine-building
//! and inference workflow.
//! \param postConfigHook Optional callback invoked after builder configuration, before engine build.
//! \return Exit code (0 for success, non-zero for failure)
[[nodiscard]] int trtexecMain(int argc, char** argv, PostConfigCallback const& postConfigHook = nullptr);

//! \brief Tuning-loop driver invoked when --tuneBuildRoutes / --tuneBuildRouteFile / --continue
//! is present in argv. Enumerates build routes per the tuning expression and search algorithm,
//! and for each iteration fork+execs trtexec with `--setBuildRoute=<route> --tuningResultFile=<json>`
//! injected. Each child runs trtexecMain() unmodified, so any iteration is reproducible by hand
//! using its --setBuildRoute argument.
//! \return Exit code (0 if at least one iteration succeeded, non-zero otherwise).
[[nodiscard]] int32_t runTuningLoop(int32_t argc, char** argv);

} // namespace sample

#endif // TRTEXEC_H
