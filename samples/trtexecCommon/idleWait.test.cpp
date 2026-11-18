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

#include "idleWait.h"
#include <atomic>
#include <deque>
#include <gtest/gtest.h>
#include <limits>

#if defined(__QNXNTO__)
#include <csignal>
#include <pthread.h>

namespace
{
struct TimeoutCall
{
    uint32_t flags;
    uint64_t requested;
    uint64_t output;
    int32_t error;
};
std::deque<TimeoutCall> gExpected;
bool gUseRealTimer{false};
std::atomic<int32_t> gInterruptions{0};

class IdleWaitTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gExpected.clear();
        gUseRealTimer = false;
    }
    void TearDown() override
    {
        EXPECT_TRUE(gExpected.empty());
    }
};
} // namespace

extern "C" int __real_TimerTimeout(clockid_t, uint32_t, struct sigevent const*, uint64_t const*, uint64_t*);

//! Interpose only in the test binary to exercise failures that cannot be induced safely on a shared board.
extern "C" int __wrap_TimerTimeout(
    clockid_t clock, uint32_t flags, struct sigevent const* notify, uint64_t const* requested, uint64_t* output)
{
    if (gUseRealTimer)
    {
        int const result = __real_TimerTimeout(clock, flags, notify, requested, output);
        if (result == -1 && errno == EINTR)
        {
            ++gInterruptions;
        }
        return result;
    }
    EXPECT_EQ(clock, CLOCK_MONOTONIC);
    EXPECT_EQ(notify, nullptr);
    if (gExpected.empty())
    {
        ADD_FAILURE() << "Unexpected TimerTimeout call";
        errno = EINVAL;
        return -1;
    }
    auto const call = gExpected.front();
    gExpected.pop_front();
    EXPECT_EQ(flags, call.flags);
    EXPECT_NE(requested, nullptr);
    if (requested)
    {
        EXPECT_EQ(*requested, call.requested);
    }
    if (output)
    {
        *output = call.output;
    }
    errno = call.error;
    return call.error ? -1 : 0;
}

namespace
{
TEST_F(IdleWaitTest, ZeroDoesNotCallKernel)
{
    EXPECT_EQ(samplesCommon::waitForIdle(0.F), 0);
}
TEST_F(IdleWaitTest, RejectsInvalidAndUnrepresentableDurations)
{
    for (auto const value : {-1.F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::max()})
    {
        EXPECT_EQ(samplesCommon::waitForIdle(value), EINVAL);
    }
}
TEST_F(IdleWaitTest, RoundsPositiveSubNanosecondUp)
{
    gExpected
        = {{TIMER_TOLERANCE, 1000, 777777, 0}, {_NTO_TIMEOUT_NANOSLEEP, 1, 0, 0}, {TIMER_TOLERANCE, 777777, 0, 0}};
    EXPECT_EQ(samplesCommon::waitForIdle(0.0000001F), 0);
}
TEST_F(IdleWaitTest, ConvertsMillisecondsAndRestoresTolerance)
{
    gExpected = {
        {TIMER_TOLERANCE, 1000, 777777, 0}, {_NTO_TIMEOUT_NANOSLEEP, 1000000, 0, 0}, {TIMER_TOLERANCE, 777777, 0, 0}};
    EXPECT_EQ(samplesCommon::waitForIdle(1.F), 0);
}
TEST_F(IdleWaitTest, InterruptedSleepRearmsToleranceAndUsesOnlyRemainingTime)
{
    gExpected = {{TIMER_TOLERANCE, 1000, 777777, 0}, {_NTO_TIMEOUT_NANOSLEEP, 1000000, 600000, EINTR},
        {TIMER_TOLERANCE, 1000, 0, 0}, {_NTO_TIMEOUT_NANOSLEEP, 600000, 200000, EINTR}, {TIMER_TOLERANCE, 1000, 0, 0},
        {_NTO_TIMEOUT_NANOSLEEP, 200000, 0, 0}, {TIMER_TOLERANCE, 777777, 0, 0}};
    EXPECT_EQ(samplesCommon::waitForIdle(1.F), 0);
}
TEST_F(IdleWaitTest, InterruptedSleepWithNoRemainingTimeOnlyRestoresTolerance)
{
    gExpected = {{TIMER_TOLERANCE, 1000, 777777, 0}, {_NTO_TIMEOUT_NANOSLEEP, 1000000, 0, EINTR},
        {TIMER_TOLERANCE, 777777, 0, 0}};
    EXPECT_EQ(samplesCommon::waitForIdle(1.F), 0);
}
TEST_F(IdleWaitTest, RearmFailureStillRestoresTolerance)
{
    gExpected = {{TIMER_TOLERANCE, 1000, 777777, 0}, {_NTO_TIMEOUT_NANOSLEEP, 1000000, 600000, EINTR},
        {TIMER_TOLERANCE, 1000, 0, EPERM}, {TIMER_TOLERANCE, 777777, 0, 0}};
    EXPECT_EQ(samplesCommon::waitForIdle(1.F), EPERM);
}
TEST_F(IdleWaitTest, PermissionFailureDoesNotSkipIntoSleep)
{
    gExpected = {{TIMER_TOLERANCE, 1000, 0, EPERM}};
    EXPECT_EQ(samplesCommon::waitForIdle(1.F), EPERM);
}
TEST_F(IdleWaitTest, SleepFailureStillRestoresTolerance)
{
    gExpected = {{TIMER_TOLERANCE, 1000, 777777, 0}, {_NTO_TIMEOUT_NANOSLEEP, 1000000, 0, EAGAIN},
        {TIMER_TOLERANCE, 777777, 0, 0}};
    EXPECT_EQ(samplesCommon::waitForIdle(1.F), EAGAIN);
}
TEST_F(IdleWaitTest, RestoreFailureIsReported)
{
    gExpected = {{TIMER_TOLERANCE, 1000, 777777, 0}, {_NTO_TIMEOUT_NANOSLEEP, 1000000, 0, 0},
        {TIMER_TOLERANCE, 777777, 0, EPERM}};
    EXPECT_EQ(samplesCommon::waitForIdle(1.F), EPERM);
}
TEST_F(IdleWaitTest, RealQnxTimerRestoresTolerance)
{
    gUseRealTimer = true;
    uint64_t before{}, after{};
    ASSERT_EQ(__real_TimerTimeout(CLOCK_MONOTONIC, TIMER_TOLERANCE, nullptr, nullptr, &before), 0);
    auto const start = std::chrono::steady_clock::now();
    ASSERT_EQ(samplesCommon::waitForIdle(0.1F), 0);
    auto const elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_GE((std::chrono::duration<double, std::micro>(elapsed).count()), 90.0);
    ASSERT_EQ(__real_TimerTimeout(CLOCK_MONOTONIC, TIMER_TOLERANCE, nullptr, nullptr, &after), 0);
    EXPECT_EQ(after, before);
}
//! A signal with no application work still interrupts a nanosleep.
void handleSignal(int) {}

TEST_F(IdleWaitTest, RealQnxSignalDoesNotTruncateIdle)
{
    gUseRealTimer = true;
    gInterruptions = 0;
    struct sigaction action
    {
    }, previous{};
    action.sa_handler = handleSignal;
    sigemptyset(&action.sa_mask);
    ASSERT_EQ(sigaction(SIGUSR1, &action, &previous), 0);
    auto const caller = pthread_self();
    std::thread interrupter([caller] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        pthread_kill(caller, SIGUSR1);
    });
    auto const start = std::chrono::steady_clock::now();
    int32_t const result = samplesCommon::waitForIdle(10.F);
    auto const elapsed = std::chrono::steady_clock::now() - start;
    interrupter.join();
    EXPECT_EQ(sigaction(SIGUSR1, &previous, nullptr), 0);
    EXPECT_EQ(result, 0);
    EXPECT_GT(gInterruptions.load(), 0);
    EXPECT_GE((std::chrono::duration<double, std::milli>(elapsed).count()), 9.0);
}
} // namespace
#else
TEST(IdleWaitTest, PortableZeroAndInvalidDurations)
{
    EXPECT_EQ(samplesCommon::waitForIdle(0.F), 0);
    EXPECT_EQ(samplesCommon::waitForIdle(-1.F), EINVAL);
    EXPECT_EQ(samplesCommon::waitForIdle(std::numeric_limits<float>::quiet_NaN()), EINVAL);
}
#endif
