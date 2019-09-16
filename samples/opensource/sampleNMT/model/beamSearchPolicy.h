/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
        int endSequenceId, LikelihoodCombinationOperator::ptr likelihoodCombinationOperator, int beamWidth);

    void initialize(int sampleCount, int* maxOutputSequenceLengths);

    void processTimestep(int validSampleCount, const float* hCombinedLikelihoods, const int* hVocabularyIndices,
        const int* hRayOptionIndices, int* hSourceRayIndices, float* hSourceLikelihoods);

    int getTailWithNoWorkRemaining();

    void readGeneratedResult(
        int sampleCount, int maxOutputSequenceLength, int* hOutputData, int* hActualOutputSequenceLengths);

    std::string getInfo() override;

    ~BeamSearchPolicy() override = default;

protected:
    struct Ray
    {
        int vocabularyId;
        int backtrackId;
    };

    void backtrack(
        int lastTimestepId, int sampleId, int lastTimestepRayId, int* hOutputData, int lastTimestepWriteId) const;

protected:
    int mEndSequenceId;
    LikelihoodCombinationOperator::ptr mLikelihoodCombinationOperator;
    int mBeamWidth;
    std::vector<bool> mValidSamples;
    std::vector<float> mCurrentLikelihoods;
    std::vector<Ray> mBeamSearchTable;
    int mSampleCount;
    std::vector<int> mMaxOutputSequenceLengths;
    int mTimestepId;

    std::vector<std::vector<int>> mCandidates;
    std::vector<float> mCandidateLikelihoods;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_BEAM_SEARCH_POLICY_
