#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (
    CompareFuncCosineSimilarityArgs,
    CompareFuncIndicesArgs,
    CompareFuncL2Args,
    CompareFuncPerceptualMetricsArgs,
    CompareFuncPsnrArgs,
    CompareFuncSimpleArgs,
    CompareFuncSnrArgs,
    ComparatorCompareArgs,
)
from polygraphy.tools.base import Tool


class Accuracy(Tool):
    """
    Re-check saved accuracy results against different thresholds.

    Loads accuracy comparison results saved by `polygraphy run --save-accuracy-results` and
    re-evaluates the stored per-output metrics against new thresholds, reporting pass/fail and
    setting the exit status accordingly -- without re-running inference.

    Because only the computed metrics are stored (not the output tensors), re-thresholding works for
    the scalar-statistic comparisons: `simple` with `--check-error-stat max|mean|median|quantile`,
    and `l2`/`cosine_similarity`/`psnr`/`snr`/`perceptual_metrics`. It does NOT work for `simple`
    with the default `elemwise` check (the per-element verdict cannot be reconstructed from summary
    statistics); for that, re-run from saved raw outputs with `run --save-outputs`/`--load-outputs`.
    """

    def __init__(self):
        super().__init__("accuracy")

    def get_subscriptions_impl(self):
        # ComparatorCompareArgs provides --compare / --check-average / --compare-func-script and
        # builds the comparison-function instances; the per-function arg groups parse the thresholds.
        # The options that require running inference or loading runs (postprocessing, --validate,
        # --load-outputs, --sequential-runners) are irrelevant here -- this subtool re-checks
        # already-computed accuracy results -- so they are disabled.
        return [
            ComparatorCompareArgs(
                allow_postprocessing=False,
                allow_validate=False,
                allow_load_outputs=False,
                allow_sequential_runners=False,
            ),
            CompareFuncSimpleArgs(),
            CompareFuncIndicesArgs(),
            CompareFuncL2Args(),
            CompareFuncCosineSimilarityArgs(),
            CompareFuncPsnrArgs(),
            CompareFuncSnrArgs(),
            CompareFuncPerceptualMetricsArgs(),
        ]

    def show_start_end_logging_impl(self, args):
        return True

    def add_parser_args_impl(self, parser):
        parser.add_argument(
            "path",
            help="Path to a saved accuracy results file "
            "(as written by `polygraphy run --save-accuracy-results`).",
        )

    def run_impl(self, args):
        from polygraphy.comparator import AccuracyResults

        compare_args = self.arg_groups[ComparatorCompareArgs]

        results = AccuracyResults.load(args.path)
        compare_funcs = compare_args.build_compare_funcs()
        aggregation = "average" if compare_args.check_average else "per_sample"

        thresholds = self._thresholds_from_funcs(compare_funcs, results.output_names())
        results.reevaluate(thresholds, aggregation=aggregation)

        self._log_summary(results)
        success = bool(results)

        if compare_args.save_accuracy_results_path:
            results.save(compare_args.save_accuracy_results_path)

        if not success:
            G_LOGGER.error(
                "FAILED | Accuracy results did not pass the requested thresholds."
            )
            return 1
        G_LOGGER.finish("PASSED | Accuracy results passed the requested thresholds.")
        return 0

    @staticmethod
    def _thresholds_from_funcs(compare_funcs, output_names):
        # Convert each comparison function into the Threshold it applies -- per-output, as an
        # {output_name: Threshold} mapping, when the saved results name their outputs. reevaluate
        # then operates purely on thresholds. Functions that produce no re-thresholdable metric
        # (e.g. 'indices' or a plain function) are skipped; a result that needed one then surfaces
        # reevaluate's clear "no matching threshold" error.
        from polygraphy.comparator.compare import metric_fields_for_func

        thresholds = []
        for func in compare_funcs:
            if not metric_fields_for_func(func):
                continue
            if output_names:
                thresholds.append(
                    {name: func.thresholds_for(name) for name in output_names}
                )
            else:
                thresholds.append(func.thresholds_for(""))
        return thresholds

    def _log_summary(self, results):
        for result in results:
            for runner_pair in result.keys():
                prefix = f"Accuracy Summary | {runner_pair[0]} vs. {runner_pair[1]} | "
                if result.aggregation == "average":
                    averaged = result.average_results(runner_pair)
                    passed = sum(bool(r) for r in averaged.values())
                    total = len(averaged)
                    G_LOGGER.log(
                        prefix + f"Passed: {passed}/{total} outputs",
                        G_LOGGER.FINISH if passed == total else G_LOGGER.ERROR,
                    )
                    with G_LOGGER.indent():
                        for output_name, (line, line_passed) in result.describe_average(
                            runner_pair
                        ).items():
                            G_LOGGER.log(
                                f"{output_name} | {line}",
                                G_LOGGER.FINISH if line_passed else G_LOGGER.ERROR,
                            )
                else:
                    matched, _, total = result.stats(runner_pair)
                    rate = (float(matched) / float(total) if total else 1.0) * 100
                    G_LOGGER.log(
                        prefix
                        + f"Passed: {matched}/{total} iterations | Pass Rate: {rate:.2f}%",
                        G_LOGGER.FINISH if matched == total else G_LOGGER.ERROR,
                    )
