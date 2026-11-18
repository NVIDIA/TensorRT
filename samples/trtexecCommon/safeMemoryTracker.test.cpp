/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "safeMemoryTracker.h"
#include <array>
#include <condition_variable>
#include <cstdint>
#include <gtest/gtest.h>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace samplesSafeCommon
{
namespace
{

TEST(SafeMemoryTracker, DisabledDoesNotRecordAllocations)
{
    SafeMemoryTracker tracker;
    int32_t allocation{};
    tracker.recordAllocation(&allocation, 64);
    bool released{};
    EXPECT_TRUE(tracker.release(&allocation, [&] {
        released = true;
        return true;
    }));
    EXPECT_TRUE(released);
    EXPECT_EQ(tracker.snapshot().peakBytes, 0U);
    EXPECT_EQ(tracker.snapshot().outstandingAllocations, 0U);
}

TEST(SafeMemoryTracker, PeakCountsSimultaneousLiveBytes)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    std::array<int32_t, 3> allocations{};
    tracker.recordAllocation(allocations.data(), 32);
    tracker.recordAllocation(&allocations[1], 64);
    EXPECT_TRUE(tracker.release(allocations.data(), [] { return true; }));
    tracker.recordAllocation(&allocations[2], 16);
    auto stats = tracker.snapshot();
    EXPECT_EQ(stats.peakBytes, 96U);
    EXPECT_EQ(stats.currentBytes, 80U);
    EXPECT_EQ(stats.outstandingAllocations, 2U);
    EXPECT_TRUE(stats.complete);
    EXPECT_TRUE(tracker.release(&allocations[1], [] { return true; }));
    EXPECT_TRUE(tracker.release(&allocations[2], [] { return true; }));
    stats = tracker.snapshot();
    EXPECT_EQ(stats.currentBytes, 0U);
    EXPECT_EQ(stats.outstandingAllocations, 0U);
    EXPECT_EQ(stats.peakBytes, 96U);
}

TEST(SafeMemoryTracker, FailedFreeRemainsOutstanding)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    int32_t allocation{};
    tracker.recordAllocation(&allocation, 64);
    EXPECT_FALSE(tracker.release(&allocation, [] { return false; }));
    EXPECT_EQ(tracker.snapshot().currentBytes, 64U);
    EXPECT_EQ(tracker.snapshot().outstandingAllocations, 1U);
    EXPECT_TRUE(tracker.release(&allocation, [] { return true; }));
    EXPECT_EQ(tracker.snapshot().currentBytes, 0U);
}

TEST(SafeMemoryTracker, NullAndUntrackedAllocationsDoNotChangeCounters)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    std::array<int32_t, 2> allocations{};
    tracker.recordAllocation(nullptr, 128);
    tracker.recordAllocation(allocations.data(), 64);
    EXPECT_TRUE(tracker.release(&allocations[1], [] { return true; }));
    auto const stats = tracker.snapshot();
    EXPECT_EQ(stats.currentBytes, 64U);
    EXPECT_EQ(stats.peakBytes, 64U);
    EXPECT_TRUE(stats.complete);
}

TEST(SafeMemoryTracker, ReusedPointerStartsANewAllocation)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    int32_t allocation{};
    tracker.recordAllocation(&allocation, 24);
    EXPECT_TRUE(tracker.release(&allocation, [] { return true; }));
    tracker.recordAllocation(&allocation, 32);
    auto const stats = tracker.snapshot();
    EXPECT_EQ(stats.currentBytes, 32U);
    EXPECT_EQ(stats.peakBytes, 32U);
    EXPECT_EQ(stats.outstandingAllocations, 1U);
    EXPECT_TRUE(stats.complete);
}

TEST(SafeMemoryTracker, NewPeakWindowRetainsLiveAllocations)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    std::array<int32_t, 2> allocations{};
    tracker.recordAllocation(allocations.data(), 40);
    tracker.recordAllocation(&allocations[1], 20);
    EXPECT_TRUE(tracker.release(&allocations[1], [] { return true; }));
    tracker.resetPeak();
    auto const stats = tracker.snapshot();
    EXPECT_EQ(stats.currentBytes, 40U);
    EXPECT_EQ(stats.peakBytes, 40U);
    EXPECT_EQ(stats.outstandingAllocations, 1U);
}

TEST(SafeMemoryTracker, CounterOverflowMarksReportIncomplete)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    std::array<int32_t, 2> allocations{};
    tracker.recordAllocation(allocations.data(), std::numeric_limits<uint64_t>::max());
    tracker.recordAllocation(&allocations[1], 1);
    EXPECT_FALSE(tracker.snapshot().complete);
    EXPECT_EQ(tracker.snapshot().currentBytes, std::numeric_limits<uint64_t>::max());
    EXPECT_TRUE(tracker.release(allocations.data(), [] { return true; }));
    EXPECT_EQ(tracker.snapshot().currentBytes, 0U);
}

TEST(SafeMemoryTracker, DuplicatePointerMarksReportIncomplete)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    int32_t allocation{};
    tracker.recordAllocation(&allocation, 64);
    tracker.recordAllocation(&allocation, 32);
    EXPECT_FALSE(tracker.snapshot().complete);
    EXPECT_EQ(tracker.snapshot().currentBytes, 64U);
}

TEST(SafeMemoryTracker, ReleaseExceptionKeepsAllocationOutstanding)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    int32_t allocation{};
    tracker.recordAllocation(&allocation, 64);
    EXPECT_THROW(
        tracker.release(&allocation, []() -> bool { throw std::runtime_error("free failed"); }), std::runtime_error);
    EXPECT_EQ(tracker.snapshot().currentBytes, 64U);
    EXPECT_TRUE(tracker.release(&allocation, [] { return true; }));
}

TEST(SafeMemoryTracker, ConcurrentWorkersShareOneHighWaterMark)
{
    constexpr size_t kWORKERS{6};
    constexpr size_t kALLOCATIONS{32};
    constexpr uint64_t kBYTES{8};
    SafeMemoryTracker tracker;
    tracker.enable();
    std::array<std::array<int32_t, kALLOCATIONS>, kWORKERS> allocations{};
    std::mutex mutex;
    std::condition_variable ready;
    size_t workersReady{};
    bool canRelease{};
    std::vector<std::thread> workers;
    for (auto& workerAllocations : allocations)
    {
        workers.emplace_back([&, items = &workerAllocations] {
            for (auto& allocation : *items)
            {
                tracker.recordAllocation(&allocation, kBYTES);
            }
            {
                std::unique_lock lock(mutex);
                ++workersReady;
                ready.notify_all();
                ready.wait(lock, [&] { return canRelease; });
            }
            for (auto& allocation : *items)
            {
                EXPECT_TRUE(tracker.release(&allocation, [] { return true; }));
            }
        });
    }
    {
        std::unique_lock lock(mutex);
        ready.wait(lock, [&] { return workersReady == kWORKERS; });
        auto const stats = tracker.snapshot();
        EXPECT_EQ(stats.currentBytes, kWORKERS * kALLOCATIONS * kBYTES);
        EXPECT_EQ(stats.outstandingAllocations, kWORKERS * kALLOCATIONS);
        canRelease = true;
        ready.notify_all();
    }
    for (auto& worker : workers)
    {
        worker.join();
    }
    auto const stats = tracker.snapshot();
    EXPECT_EQ(stats.peakBytes, kWORKERS * kALLOCATIONS * kBYTES);
    EXPECT_EQ(stats.currentBytes, 0U);
    EXPECT_EQ(stats.outstandingAllocations, 0U);
    EXPECT_TRUE(stats.complete);
}

TEST(SafeMemoryTracker, PlacementAndPurposePeaksAreIndependent)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    std::array<int32_t, 3> allocations{};
    using Kind = SafeMemoryTracker::Kind;
    using Purpose = SafeMemoryTracker::Purpose;
    tracker.recordAllocation(allocations.data(), 100, Kind::kGPU, Purpose::kIMMUTABLE);
    tracker.recordAllocation(&allocations[1], 20, Kind::kPINNED, Purpose::kIO);
    EXPECT_TRUE(tracker.release(allocations.data(), [] { return true; }));
    tracker.recordAllocation(&allocations[2], 60, Kind::kCPU, Purpose::kSCRATCH);
    auto const stats = tracker.snapshot();
    EXPECT_EQ(stats.peakBytes, 120U);
    EXPECT_EQ(stats.currentBytes, 80U);
    EXPECT_EQ(stats.byKind[static_cast<size_t>(Kind::kGPU)].peakBytes, 100U);
    EXPECT_EQ(stats.byKind[static_cast<size_t>(Kind::kGPU)].currentBytes, 0U);
    EXPECT_EQ(stats.byKind[static_cast<size_t>(Kind::kPINNED)].currentBytes, 20U);
    EXPECT_EQ(stats.byKind[static_cast<size_t>(Kind::kCPU)].currentBytes, 60U);
    EXPECT_EQ(stats.byPurpose[static_cast<size_t>(Purpose::kIMMUTABLE)].peakBytes, 100U);
    EXPECT_EQ(stats.byPurpose[static_cast<size_t>(Purpose::kSCRATCH)].peakBytes, 60U);
    EXPECT_EQ(stats.byPurpose[static_cast<size_t>(Purpose::kIO)].peakBytes, 20U);
}

TEST(SafeMemoryTracker, CategoryResetRetainsLiveAllocations)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    int32_t allocation{};
    using Kind = SafeMemoryTracker::Kind;
    using Purpose = SafeMemoryTracker::Purpose;
    tracker.recordAllocation(&allocation, 40, Kind::kGPU, Purpose::kSCRATCH);
    EXPECT_TRUE(tracker.release(&allocation, [] { return true; }));
    tracker.recordAllocation(&allocation, 20, Kind::kGPU, Purpose::kSCRATCH);
    tracker.resetPeak();
    auto const stats = tracker.snapshot();
    EXPECT_EQ(stats.byKind[static_cast<size_t>(Kind::kGPU)].peakBytes, 20U);
    EXPECT_EQ(stats.byPurpose[static_cast<size_t>(Purpose::kSCRATCH)].peakBytes, 20U);
    EXPECT_EQ(stats.byPurpose[static_cast<size_t>(Purpose::kSCRATCH)].outstandingAllocations, 1U);
}

TEST(SafeMemoryTracker, ReleaseCheckFindsRetainedAllocations)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    auto const before = tracker.checkpoint();
    std::array<int32_t, 2> allocations{};
    tracker.recordAllocation(allocations.data(), 32);
    tracker.recordAllocation(&allocations[1], 64);
    EXPECT_TRUE(tracker.release(allocations.data(), [] { return true; }));
    auto const result = tracker.checkReleased(before);
    EXPECT_EQ(result.outstandingBytes, 64U);
    EXPECT_EQ(result.outstandingAllocations, 1U);
    EXPECT_TRUE(result.complete);
}

TEST(SafeMemoryTracker, ReleaseCheckIgnoresPreexistingAllocations)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    std::array<int32_t, 2> allocations{};
    tracker.recordAllocation(allocations.data(), 64);
    auto const before = tracker.checkpoint();
    tracker.recordAllocation(&allocations[1], 32);
    EXPECT_TRUE(tracker.release(&allocations[1], [] { return true; }));
    auto const result = tracker.checkReleased(before);
    EXPECT_EQ(result.outstandingAllocations, 0U);
    EXPECT_EQ(tracker.snapshot().currentBytes, 64U);
    EXPECT_TRUE(result.complete);
}

TEST(SafeMemoryTracker, EqualTotalsCannotHideAReplacedBaselineAllocation)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    int32_t allocation{};
    tracker.recordAllocation(&allocation, 64);
    auto const before = tracker.checkpoint();
    EXPECT_TRUE(tracker.release(&allocation, [] { return true; }));
    tracker.recordAllocation(&allocation, 64);
    auto const result = tracker.checkReleased(before);
    EXPECT_EQ(result.outstandingBytes, 64U);
    EXPECT_EQ(result.outstandingAllocations, 1U);
    EXPECT_TRUE(result.complete);
}

TEST(SafeMemoryTracker, FailedFreeIsIncludedInReleaseCheck)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    auto const before = tracker.checkpoint();
    int32_t allocation{};
    tracker.recordAllocation(&allocation, 64);
    EXPECT_FALSE(tracker.release(&allocation, [] { return false; }));
    EXPECT_EQ(tracker.checkReleased(before).outstandingAllocations, 1U);
    EXPECT_TRUE(tracker.release(&allocation, [] { return true; }));
    EXPECT_EQ(tracker.checkReleased(before).outstandingAllocations, 0U);
}

TEST(SafeMemoryTracker, IncompleteAccountingCannotPassReleaseCheck)
{
    SafeMemoryTracker tracker;
    EXPECT_FALSE(tracker.checkReleased(tracker.checkpoint()).complete);
    tracker.enable();
    auto const before = tracker.checkpoint();
    int32_t allocation{};
    tracker.recordAllocation(&allocation, 64);
    tracker.recordAllocation(&allocation, 32);
    EXPECT_TRUE(tracker.release(&allocation, [] { return true; }));
    auto const result = tracker.checkReleased(before);
    EXPECT_EQ(result.outstandingAllocations, 0U);
    EXPECT_FALSE(result.complete);
}

TEST(SafeMemoryTracker, RepeatedCyclesReleaseAllNewAllocations)
{
    SafeMemoryTracker tracker;
    tracker.enable();
    int32_t allocation{};
    for (int32_t cycle = 0; cycle < 10; ++cycle)
    {
        auto const before = tracker.checkpoint();
        tracker.recordAllocation(&allocation, 64);
        EXPECT_TRUE(tracker.release(&allocation, [] { return true; }));
        auto const result = tracker.checkReleased(before);
        EXPECT_EQ(result.outstandingAllocations, 0U);
        EXPECT_TRUE(result.complete);
        tracker.resetPeak();
        EXPECT_EQ(tracker.snapshot().peakBytes, 0U);
    }
}

} // namespace
} // namespace samplesSafeCommon
