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

#include "bleuScoreWriter.h"
#include "logger.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace nmtSample
{

typedef std::vector<std::string> Segment_t;
typedef std::map<Segment_t, int> Count_t;
int read(std::vector<Segment_t>& samples, std::shared_ptr<std::istream> input, int samplesToRead = 1)
{
    std::string line;
    int lineCounter = 0;
    Segment_t tokens;
    samples.resize(0);
    std::string pattern("@@ ");
    while (lineCounter < samplesToRead && std::getline(*input, line))
    {
        // if clean and handle BPE or SPM outputs is required
        std::size_t p0 = 0;
        while ((p0 = line.find(pattern, p0)) != std::string::npos)
        {
            line.replace(p0, pattern.length(), "");
        }

        // generate error if those special characters exist. Windows needs explicit encoding.
#ifdef _MSC_VER
        p0 = line.find(u8"\u2581");
#else
        p0 = line.find("\u2581");
#endif
        assert((p0 == std::string::npos));
        std::istringstream ss(line);
        std::string token;
        tokens.resize(0);
        while (ss >> token)
        {
            tokens.emplace_back(token);
        }
        samples.emplace_back(tokens);
        lineCounter++;
    }
    return lineCounter;
}

Count_t ngramCounts(const Segment_t& segment, int maxOrder = 4)
{
    Count_t ngramCounts;

    for (int order = 1; order < maxOrder + 1; order++)
    {
        for (int i = 0; i < static_cast<int>(segment.size()) - order + 1; i++)
        {
            Segment_t ngram;
            for (int j = i; j < i + order; j++)
                ngram.emplace_back(segment[j]);

            auto it = ngramCounts.find(ngram);
            if (it != ngramCounts.end())
            {
                it->second++;
            }
            else
                ngramCounts[ngram] = 1;
        }
    }

    return ngramCounts;
}

Count_t ngramCountIntersection(const Count_t& cnt0, const Count_t& cnt1)
{
    Count_t overlap;
    // merge the maps
    auto it0 = cnt0.begin(), it1 = cnt1.begin(), end0 = cnt0.end(), end1 = cnt1.end();
    while (it0 != end0 && it1 != end1)
    {
        if (it0->first == it1->first)
        {
            overlap.emplace(it0->first, std::min(it0->second, it1->second));
            it0++;
            it1++;
        }
        else
        {
            if (it0->first < it1->first)
                it0++;
            else
                it1++;
        }
    }
    return overlap;
}

void accumulateBLEU(const std::vector<Segment_t>& referenceSamples, const std::vector<Segment_t>& outputSamples,
    int maxOrder, size_t& referenceLength, size_t& translationLength, std::vector<size_t>& matchesByOrder,
    std::vector<size_t>& possibleMatchesByOrder)
{
    assert(referenceSamples.size() == outputSamples.size());
    auto reference = referenceSamples.begin();
    auto translation = outputSamples.begin();

    while (translation != outputSamples.end())
    {
        referenceLength += reference->size();
        translationLength += translation->size();

        Count_t refNgramCounts = ngramCounts(*reference);
        Count_t outputNgramCounts = ngramCounts(*translation);
        Count_t overlap = ngramCountIntersection(outputNgramCounts, refNgramCounts);
        for (auto& ngram : overlap)
        {
            matchesByOrder[ngram.first.size() - 1] += ngram.second;
        }
        for (int order = 1; order < maxOrder + 1; order++)
        {
            int possibleMatches = static_cast<int>(translation->size()) - order + 1;
            if (possibleMatches > 0)
                possibleMatchesByOrder[order - 1] += possibleMatches;
        }
        ++translation;
        ++reference;
    }
}

BLEUScoreWriter::BLEUScoreWriter(
    std::shared_ptr<std::istream> referenceTextInput, Vocabulary::ptr vocabulary, int maxOrder)
    : mReferenceInput(referenceTextInput)
    , mVocabulary(vocabulary)
    , mReferenceLength(0)
    , mTranslationLength(0)
    , mMaxOrder(maxOrder)
    , mSmooth(false)
    , mMatchesByOrder(maxOrder, 0)
    , mPossibleMatchesByOrder(maxOrder, 0)
{
}

void BLEUScoreWriter::write(const int* hOutputData, int actualOutputSequenceLength, int actualInputSequenceLength)
{
    std::vector<Segment_t> outputSamples;
    std::vector<Segment_t> referenceSamples;
    int numReferenceSamples = read(referenceSamples, mReferenceInput, 1);
    assert(numReferenceSamples == 1);

    Segment_t segment;
    std::stringstream filteredSentence(DataWriter::generateText(actualOutputSequenceLength, hOutputData, mVocabulary));
    std::string token;
    while (filteredSentence >> token)
    {
        segment.emplace_back(token);
    }
    outputSamples.emplace_back(segment);

    accumulateBLEU(referenceSamples, outputSamples, mMaxOrder, mReferenceLength, mTranslationLength, mMatchesByOrder,
        mPossibleMatchesByOrder);
}

void BLEUScoreWriter::initialize() {}

void BLEUScoreWriter::finalize()
{
    gLogInfo << "BLEU score = " << getScore() << std::endl;
}

float BLEUScoreWriter::getScore() const
{
    std::vector<double> precisions(mMaxOrder, 0.0);
    for (int i = 0; i < mMaxOrder; i++)
    {
        if (mSmooth)
        {
            precisions[i] = ((mMatchesByOrder[i] + 1.) / (mPossibleMatchesByOrder[i] + 1.));
        }
        else
        {
            if (mPossibleMatchesByOrder[i] > 0)
                precisions[i] = (static_cast<double>(mMatchesByOrder[i]) / mPossibleMatchesByOrder[i]);
            else
                precisions[i] = 0.0;
        }
    }
    double pLogSum, geoMean;
    if (*std::min_element(precisions.begin(), precisions.end()) > 0.0)
    {
        pLogSum = 0.0;
        for (auto p : precisions)
            pLogSum += (1. / mMaxOrder) * log(p);
        geoMean = exp(pLogSum);
    }
    else
        geoMean = 0.0;

    double ratio = static_cast<double>(mTranslationLength) / mReferenceLength;
    double bp;
    bp = (ratio > 1.0) ? 1.0 : exp(1.0 - 1.0 / ratio);
    return static_cast<float>(geoMean * bp * 100.0);
}

std::string BLEUScoreWriter::getInfo()
{
    std::stringstream ss;
    ss << "BLEU Score Writer, max order = " << mMaxOrder;
    return ss.str();
}
} // namespace nmtSample
