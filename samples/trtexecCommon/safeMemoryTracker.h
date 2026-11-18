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

#ifndef SAFE_MEMORY_TRACKER_H
#define SAFE_MEMORY_TRACKER_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace samplesSafeCommon
{

//! Counts requested allocation bytes; excludes driver memory and allocator overhead.
class SafeMemoryTracker
{
public:
    enum class Kind : size_t
    {
        kCPU,
        kPINNED,
        kGPU,
        kUNKNOWN
    };

    enum class Purpose : size_t
    {
        kOTHER,
        kIMMUTABLE,
        kSCRATCH,
        kIO
    };

    struct Counters
    {
        uint64_t currentBytes{};
        uint64_t peakBytes{};
        size_t outstandingAllocations{};
    };

    //! Category peaks are independent high-water marks, not additive contributions to the total peak.
    struct Snapshot : Counters
    {
        bool complete{true};
        std::array<Counters, 4> byKind{};
        std::array<Counters, 4> byPurpose{};
    };

    //! Allocation identities distinguish a newly leaked allocation from a released baseline allocation.
    struct Checkpoint
    {
        uint64_t nextAllocation{};
        bool complete{true};
    };

    struct ReleaseCheck
    {
        uint64_t outstandingBytes{};
        size_t outstandingAllocations{};
        bool complete{true};
    };

    //! Enables tracking before inference worker threads start.
    void enable() noexcept
    {
        mEnabled.store(true, std::memory_order_relaxed);
    }

    //! Records a successful allocation; bookkeeping failure does not fail inference.
    void recordAllocation(
        void* memory, uint64_t size, Kind kind = Kind::kUNKNOWN, Purpose purpose = Purpose::kOTHER) noexcept
    {
        if (!mEnabled.load(std::memory_order_relaxed) || memory == nullptr)
        {
            return;
        }
        try
        {
            std::lock_guard lock(mMutex);
            if (size > std::numeric_limits<uint64_t>::max() - mStats.currentBytes
                || mNextAllocation == std::numeric_limits<uint64_t>::max()
                || static_cast<size_t>(kind) >= mStats.byKind.size()
                || static_cast<size_t>(purpose) >= mStats.byPurpose.size()
                || !mAllocations.emplace(memory, Allocation{size, mNextAllocation, kind, purpose}).second)
            {
                mComplete.store(false, std::memory_order_relaxed);
                return;
            }
            ++mNextAllocation;
            add(mStats, size);
            add(mStats.byKind[static_cast<size_t>(kind)], size);
            add(mStats.byPurpose[static_cast<size_t>(purpose)], size);
        }
        catch (...)
        {
            mComplete.store(false, std::memory_order_relaxed);
        }
    }

    //! Removes an allocation only after release succeeds. Serializes pointer reuse with recording.
    template <typename Release>
    bool release(void* memory, Release&& releaseMemory)
    {
        if (!mEnabled.load(std::memory_order_relaxed))
        {
            return releaseMemory();
        }
        std::lock_guard lock(mMutex);
        if (!releaseMemory())
        {
            return false;
        }
        auto const it = mAllocations.find(memory);
        if (it != mAllocations.end())
        {
            auto const& allocation = it->second;
            remove(mStats, allocation.size);
            remove(mStats.byKind[static_cast<size_t>(allocation.kind)], allocation.size);
            remove(mStats.byPurpose[static_cast<size_t>(allocation.purpose)], allocation.size);
            mAllocations.erase(it);
        }
        return true;
    }

    //! Starts a new peak window, retaining any allocations still outstanding.
    void resetPeak()
    {
        std::lock_guard lock(mMutex);
        mStats.peakBytes = mStats.currentBytes;
        for (auto& counters : mStats.byKind)
        {
            counters.peakBytes = counters.currentBytes;
        }
        for (auto& counters : mStats.byPurpose)
        {
            counters.peakBytes = counters.currentBytes;
        }
    }

    //! Returns a snapshot safe to read after concurrent allocation callbacks.
    [[nodiscard]] Snapshot snapshot() const
    {
        std::lock_guard lock(mMutex);
        auto stats = mStats;
        stats.complete = mComplete.load(std::memory_order_relaxed);
        return stats;
    }

    //! Marks the start of a create/infer/destroy cycle, preserving pre-existing allocations.
    [[nodiscard]] Checkpoint checkpoint() const
    {
        std::lock_guard lock(mMutex);
        return {mNextAllocation, mEnabled.load(std::memory_order_relaxed) && mComplete.load(std::memory_order_relaxed)};
    }

    //! Finds allocations made since the checkpoint that remain live after cleanup.
    [[nodiscard]] ReleaseCheck checkReleased(Checkpoint const& before) const
    {
        std::lock_guard lock(mMutex);
        ReleaseCheck result{};
        result.complete = before.complete && mComplete.load(std::memory_order_relaxed);
        for (auto const& [memory, allocation] : mAllocations)
        {
            if (allocation.id >= before.nextAllocation)
            {
                result.outstandingBytes += allocation.size;
                ++result.outstandingAllocations;
            }
        }
        return result;
    }

private:
    struct Allocation
    {
        uint64_t size;
        uint64_t id;
        Kind kind;
        Purpose purpose;
    };

    //! Update the total and category counters while holding the tracker mutex.
    static void add(Counters& counters, uint64_t size) noexcept
    {
        counters.currentBytes += size;
        counters.peakBytes = std::max(counters.peakBytes, counters.currentBytes);
        ++counters.outstandingAllocations;
    }

    //! Release exactly the categories recorded at allocation time.
    static void remove(Counters& counters, uint64_t size) noexcept
    {
        counters.currentBytes -= size;
        --counters.outstandingAllocations;
    }

    std::atomic<bool> mEnabled{false};
    std::atomic<bool> mComplete{true};
    mutable std::mutex mMutex;
    std::unordered_map<void*, Allocation> mAllocations;
    Snapshot mStats{};
    uint64_t mNextAllocation{};
};

} // namespace samplesSafeCommon

#endif // SAFE_MEMORY_TRACKER_H
