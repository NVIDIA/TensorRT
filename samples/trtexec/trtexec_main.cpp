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

#include "sampleUtils.h" // for sample::peekArg
#include "trtexec.h"

//! Tuning vs single-run dispatch. The peek is intentionally minimal — a real
//! parse would discard the original argv before runTuningLoop can re-use it
//! for its child fork+execs. Each branch does its own parseArgs.
int main(int argc, char** argv)
{
    if (sample::peekArg(argc, argv, "--tuneBuildRoutes")
        || sample::peekArg(argc, argv, "--tuneBuildRouteFile")
        || sample::peekArg(argc, argv, "--continue"))
    {
        return sample::runTuningLoop(argc, argv);
    }
    return sample::trtexecMain(argc, argv);
}
