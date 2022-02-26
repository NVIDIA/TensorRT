/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "beamSearchPolicy.h"
#include "common.h"
#ifdef _MSC_VER
// Macro definition needed to avoid name collision with std::min/max and Windows.h min/max
#define NOMINMAX
#endif
#include <algorithm>
#include <limits>
#include <sstream>

namespace nmtSample
{
BeamSearchPolicy::BeamSearchPolicy(
    int32_t endSequenceId, LikelihoodCombinationOperator::ptr& likelihoodCombinationOperator, int32_t beamWidth)
    : mEndSequenceId(endSequenceId)
    , mLikelihoodCombinationOperator(likelihoodCombinationOperator)
    , mBeamWidth(beamWidth)
    , mSampleCount(0)
    , mTimestepId(0)
{
}

void BeamSearchPolicy::initialize(int32_t sampleCount, int32_t* maxOutputSequenceLengths)
{
    mSampleCount = sampleCount;
    mMaxOutputSequenceLengths.resize(mSampleCount);
    std::copy(maxOutputSequenceLengths, maxOutputSequenceLengths + mSampleCount, &mMaxOutputSequenceLengths[0]);

    mValidSamples.resize(mSampleCount);
    std::fill(mValidSamples.begin(), mValidSamples.end(), true);

    mCurrentLikelihoods.resize(mSampleCount * mBeamWidth);
    std::fill(mCurrentLikelihoods.begin(), mCurrentLikelihoods.end(), mLikelihoodCombinationOperator->init());

    mBeamSearchTable.clear();

    mTimestepId = 0;

    mCandidates.resize(mSampleCount);
    mCandidateLikelihoods.resize(mSampleCount);
    std::fill(mCandidateLikelihoods.begin(), mCandidateLikelihoods.end(),
        mLikelihoodCombinationOperator->smallerThanMinimalLikelihood());
}

void BeamSearchPolicy::processTimestep(int32_t validSampleCount, const float* hCombinedLikelihoods,
    const int32_t* hVocabularyIndices, const int32_t* hRayOptionIndices, int32_t* hSourceRayIndices,
    float* hSourceLikelihoods)
{
    ++mTimestepId;
    mBeamSearchTable.resize(mTimestepId * mSampleCount * mBeamWidth);
    auto baseBeamSearchTable = mBeamSearchTable.begin() + (mTimestepId - 1) * mSampleCount * mBeamWidth;

    for (int32_t sampleId = 0; sampleId < validSampleCount; ++sampleId)
    {
        auto currentSourceRayIndices = hSourceRayIndices + sampleId * mBeamWidth;
        auto currentLikelihoods = hSourceLikelihoods + sampleId * mBeamWidth;
        auto currentBeamSearchTable = baseBeamSearchTable + sampleId * mBeamWidth;

        int32_t rayId = 0;
        if (mValidSamples[sampleId])
        {
            for (; rayId < mBeamWidth; ++rayId)
            {
                float optionCombinedLikelihood = hCombinedLikelihoods[sampleId * mBeamWidth + rayId];

                // Check if the current candidate is already better than this option
                if (optionCombinedLikelihood <= mCandidateLikelihoods[sampleId])
                    break; // The remaining options are even worse

                int32_t optionOriginalRayId = hRayOptionIndices[sampleId * mBeamWidth + rayId] / mBeamWidth;
                int32_t optionVocabularyId = hVocabularyIndices[sampleId * mBeamWidth + rayId];

                if ((optionVocabularyId == mEndSequenceId) || (mTimestepId >= mMaxOutputSequenceLengths[sampleId]))
                {
                    // We have a new candidate output sequence for the sample
                    mCandidateLikelihoods[sampleId] = optionCombinedLikelihood;
                    auto& candidate = mCandidates[sampleId];
                    candidate.resize(mTimestepId);
                    backtrack(mTimestepId - 2, sampleId, optionOriginalRayId, &candidate[0], mTimestepId - 2);
                    candidate[mTimestepId - 1] = optionVocabularyId;
                    break;
                }

                *(currentSourceRayIndices + rayId) = optionOriginalRayId;
                *(currentLikelihoods + rayId) = optionCombinedLikelihood;
                (currentBeamSearchTable + rayId)->vocabularyId = optionVocabularyId;
                (currentBeamSearchTable + rayId)->backtrackId = optionOriginalRayId;
            }

            // No valid rays left for the sample
            if (rayId == 0)
                mValidSamples[sampleId] = false;
        }

        // Mark the remaining rays as invalid ones
        for (; rayId < mBeamWidth; ++rayId)
        {
            *(currentSourceRayIndices + rayId) = 0;
            *(currentLikelihoods + rayId) = mLikelihoodCombinationOperator->smallerThanMinimalLikelihood();
            (currentBeamSearchTable + rayId)->vocabularyId = mEndSequenceId;
            (currentBeamSearchTable + rayId)->backtrackId = 0;
        }
    }
}

int32_t BeamSearchPolicy::getTailWithNoWorkRemaining()
{
    for (int32_t sampleId = mSampleCount - 1; sampleId >= 0; --sampleId)
    {
        if (mValidSamples[sampleId])
            return sampleId + 1;
    }
    return 0;
}

void BeamSearchPolicy::readGeneratedResult(
    int32_t sampleCount, int32_t maxOutputSequenceLength, int32_t* hOutputData, int32_t* hActualOutputSequenceLengths)
{
    for (int32_t sampleId = 0; sampleId < sampleCount; ++sampleId)
    {
        if (mCandidateLikelihoods[sampleId] > mLikelihoodCombinationOperator->smallerThanMinimalLikelihood())
        {
            // We have a candidate (finished sequence)
            std::copy_n(mCandidates[sampleId].begin(),
                std::min(static_cast<int32_t>(mCandidates[sampleId].size()), maxOutputSequenceLength),
                hOutputData + sampleId * maxOutputSequenceLength);
            hActualOutputSequenceLengths[sampleId] = mCandidates[sampleId].size();
        }
        else
        {
            // We don't have a finished sequence generated, will output the unfinished one with the highest likelihood
            ASSERT(mValidSamples[sampleId]);
            backtrack(mTimestepId - 1, sampleId, 0, hOutputData + sampleId * maxOutputSequenceLength,
                maxOutputSequenceLength - 1);
            hActualOutputSequenceLengths[sampleId] = mTimestepId;
        }
    }
}

void BeamSearchPolicy::backtrack(int32_t lastTimestepId, int32_t sampleId, int32_t lastTimestepRayId,
    int32_t* hOutputData, int32_t lastTimestepWriteId) const
{
    int32_t rayId = lastTimestepRayId;
    for (int32_t timestepId = lastTimestepId; timestepId >= 0; --timestepId)
    {
        const auto& entry = mBeamSearchTable[(timestepId * mSampleCount + sampleId) * mBeamWidth + rayId];
        rayId = entry.backtrackId;
        if (timestepId <= lastTimestepWriteId)
            hOutputData[timestepId] = entry.vocabularyId;
    }
}

std::string BeamSearchPolicy::getInfo()
{
    std::stringstream ss;
    ss << "Beam Search Policy, beam = " << mBeamWidth;
    return ss.str();
}
} // namespace nmtSample
