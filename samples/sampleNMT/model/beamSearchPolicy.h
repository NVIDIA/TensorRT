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

#ifndef SAMPLE_NMT_BEAM_SEARCH_POLICY_
#define SAMPLE_NMT_BEAM_SEARCH_POLICY_

#include "../component.h"
#include "likelihoodCombinationOperator.h"

#include <vector>

namespace nmtSample
{
/** \class BeamSearchPolicy
 *
 * \brief processes the results of one iteration of the generator with beam search and produces input for the next
 * iteration
 *
 */
class BeamSearchPolicy : public Component
{
public:
    typedef std::shared_ptr<BeamSearchPolicy> ptr;

    BeamSearchPolicy(
        int32_t endSequenceId, LikelihoodCombinationOperator::ptr& likelihoodCombinationOperator, int32_t beamWidth);

    void initialize(int32_t sampleCount, int32_t* maxOutputSequenceLengths);

    void processTimestep(int32_t validSampleCount, const float* hCombinedLikelihoods, const int32_t* hVocabularyIndices,
        const int32_t* hRayOptionIndices, int32_t* hSourceRayIndices, float* hSourceLikelihoods);

    int32_t getTailWithNoWorkRemaining();

    void readGeneratedResult(int32_t sampleCount, int32_t maxOutputSequenceLength, int32_t* hOutputData,
        int32_t* hActualOutputSequenceLengths);

    std::string getInfo() override;

    ~BeamSearchPolicy() override = default;

protected:
    struct Ray
    {
        int32_t vocabularyId;
        int32_t backtrackId;
    };

    void backtrack(int32_t lastTimestepId, int32_t sampleId, int32_t lastTimestepRayId, int32_t* hOutputData,
        int32_t lastTimestepWriteId) const;

protected:
    int32_t mEndSequenceId;
    LikelihoodCombinationOperator::ptr mLikelihoodCombinationOperator;
    int32_t mBeamWidth;
    std::vector<bool> mValidSamples;
    std::vector<float> mCurrentLikelihoods;
    std::vector<Ray> mBeamSearchTable;
    int32_t mSampleCount;
    std::vector<int32_t> mMaxOutputSequenceLengths;
    int32_t mTimestepId;

    std::vector<std::vector<int32_t>> mCandidates;
    std::vector<float> mCandidateLikelihoods;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_BEAM_SEARCH_POLICY_
