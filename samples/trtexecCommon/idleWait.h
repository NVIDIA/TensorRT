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

#ifndef TRT_TOOLS_TRTEXEC_COMMON_IDLE_WAIT_H
#define TRT_TOOLS_TRTEXEC_COMMON_IDLE_WAIT_H

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>

#if defined(__QNXNTO__)
#include <sys/neutrino.h>
#endif

namespace samplesCommon
{
//! \brief Sleep between two inference requests (the --idleTime pacing of trtexec and trtexec_safe).
//!
//! On QNX, std::this_thread::sleep_for() ends up in TimerTimeout() with the thread's default timeout
//! tolerance, so the wake-up is aligned to the next system tick (1 ms by default). A sub-millisecond idle
//! therefore always lasts until the next tick boundary and the request loop is pinned to one request per
//! tick (~1000 QPS). This helper asks QNX for a high-resolution timeout (1 us tolerance) for the duration of
//! the wait and restores the previous tolerance afterwards; the tolerance is consumed by each timeout, so it
//! is re-armed before retrying after a signal. On other platforms it is a plain sleep_for().
//!
//! \param milliseconds Requested idle time; must be finite and non-negative. Zero returns immediately.
//! \return 0 on success, or an errno value (EINVAL for a bad argument, EPERM when the calling thread lacks
//!         PROCMGR_AID_HIGH_RESOLUTION_TIMER on QNX, or the error reported by TimerTimeout()).
inline int32_t waitForIdle(float milliseconds)
{
    if (!std::isfinite(milliseconds) || milliseconds < 0.F)
    {
        return EINVAL;
    }
    if (milliseconds == 0.F)
    {
        return 0;
    }
#if defined(__QNXNTO__)
    // Round up so that a positive sub-nanosecond request never becomes a zero wait.
    double const nanoseconds = std::ceil(static_cast<double>(milliseconds) * 1000000.0);
    if (nanoseconds >= static_cast<double>(std::numeric_limits<uint64_t>::max()))
    {
        return EINVAL;
    }
    uint64_t remaining = static_cast<uint64_t>(nanoseconds);
    constexpr uint64_t kTOLERANCE_NS{1000U};
    uint64_t previousTolerance{};
    if (TimerTimeout(CLOCK_MONOTONIC, TIMER_TOLERANCE, nullptr, &kTOLERANCE_NS, &previousTolerance) == -1)
    {
        return errno;
    }
    int32_t error{0};
    while (remaining > 0)
    {
        uint64_t const requested = remaining;
        if (TimerTimeout(CLOCK_MONOTONIC, _NTO_TIMEOUT_NANOSLEEP, nullptr, &requested, &remaining) == 0)
        {
            break;
        }
        if (errno != EINTR)
        {
            error = errno;
            break;
        }
        // The tolerance applies to one timeout; a signal consumes it along with that timeout.
        if (remaining > 0 && TimerTimeout(CLOCK_MONOTONIC, TIMER_TOLERANCE, nullptr, &kTOLERANCE_NS, nullptr) == -1)
        {
            error = errno;
            break;
        }
    }
    if (TimerTimeout(CLOCK_MONOTONIC, TIMER_TOLERANCE, nullptr, &previousTolerance, nullptr) == -1)
    {
        return errno;
    }
    return error;
#else
    std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(milliseconds));
    return 0;
#endif
}
} // namespace samplesCommon

#endif // TRT_TOOLS_TRTEXEC_COMMON_IDLE_WAIT_H
