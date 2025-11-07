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
import functools
import math
import os

from polygraphy import config, mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.datatype import DataType

import math
import os

np = mod.lazy_import("numpy")
plt = mod.lazy_import("matplotlib.pyplot")
matplotlib = mod.lazy_import("matplotlib")


def cast_up(buffer):
    dtype = util.array.dtype(buffer)

    if dtype == DataType.FLOAT16:
        buffer = util.array.cast(buffer, DataType.FLOAT32)
    elif dtype in [DataType.INT8, DataType.UINT8, DataType.INT16, DataType.UINT16]:
        buffer = util.array.cast(buffer, DataType.INT32)
    elif dtype == DataType.UINT32:
        buffer = util.array.cast(buffer, DataType.INT64)
    return buffer


def use_higher_precision(func):
    """
    Decorator that will cast the input numpy buffer(s) to a higher precision before computation.
    """

    @functools.wraps(func)
    def wrapped(*buffers):
        if any(util.is_empty_shape(util.array.shape(buffer)) for buffer in buffers):
            return 0

        new_buffers = [cast_up(buffer) for buffer in buffers]
        return func(*new_buffers)

    return wrapped


@use_higher_precision
def compute_max(buffer):
    return util.array.max(buffer)


# Returns index of max value
@use_higher_precision
def compute_argmax(buffer):
    return util.array.unravel_index(util.array.argmax(buffer), util.array.shape(buffer))


@use_higher_precision
def compute_min(buffer):
    return util.array.min(buffer)


# Returns index of min value
@use_higher_precision
def compute_argmin(buffer):
    return util.array.unravel_index(util.array.argmin(buffer), util.array.shape(buffer))


def compute_mean(buffer):
    return util.array.mean(buffer, dtype=DataType.FLOAT32)


@use_higher_precision
def compute_std(buffer):
    return util.array.std(buffer)


@use_higher_precision
def compute_variance(buffer):
    return util.array.var(buffer)


@use_higher_precision
def compute_median(buffer):
    return util.array.median(buffer)


def compute_quantile(buffer, q):
    return util.array.quantile(buffer, q)


def compute_average_magnitude(buffer):
    return util.array.mean(util.array.abs(buffer), dtype=DataType.FLOAT32)


def str_histogram(output, hist_range=None):
    if util.array.dtype(output) == DataType.BOOL:
        return ""

    try:
        try:
            hist, bin_edges = util.array.histogram(output, range=hist_range)
        except (ValueError, RuntimeError) as err:
            G_LOGGER.verbose(f"Could not generate histogram. Note: Error was: {err}")
            return ""

        max_num_elems = compute_max(hist)
        if not max_num_elems:  # Empty tensor
            return

        bin_edges = [f"{bin:.3g}" for bin in bin_edges]
        max_start_bin_width = max(len(bin) for bin in bin_edges)
        max_end_bin_width = max(len(bin) for bin in bin_edges[1:])

        MAX_WIDTH = 40
        ret = "---- Histogram ----\n"
        ret += f"{'Bin Range':{max_start_bin_width + max_end_bin_width + 5}}|  Num Elems | Visualization\n"
        for num, bin_start, bin_end in zip(hist, bin_edges, bin_edges[1:]):
            bar = "#" * int(MAX_WIDTH * float(num) / float(max_num_elems))
            ret += f"({bin_start:<{max_start_bin_width}}, {bin_end:<{max_end_bin_width}}) | {num:10} | {bar}\n"
        return ret
    except Exception as err:
        G_LOGGER.verbose(f"Could not generate histogram.\nNote: Error was: {err}")
        if config.INTERNAL_CORRECTNESS_CHECKS:
            raise
        return ""


def str_output_stats(output, runner_name=None):
    ret = ""
    if runner_name:
        ret += f"{runner_name} | Stats: "

    try:
        ret += f"mean={compute_mean(output):.5g}, std-dev={compute_std(output):.5g}, var={compute_variance(output):.5g}, median={compute_median(output):.5g}, min={compute_min(output):.5g} at {compute_argmin(output)}, max={compute_max(output):.5g} at {compute_argmax(output)}, avg-magnitude={compute_average_magnitude(output):.5g}"

        # np.quantile doesn't work with boolean input, so we don't show quantile error if the output type is boolean
        if output.dtype == bool:
            ret += "\n"
        else:
            ret += f", p90={compute_quantile(output, 0.9):.5g}, p95={compute_quantile(output, 0.95):.5g}, p99={compute_quantile(output, 0.99):.5g}\n"
    except Exception as err:
        G_LOGGER.verbose(f"Could not generate statistics.\nNote: Error was: {err}")
        ret += "<Error while computing statistics>"
        if config.INTERNAL_CORRECTNESS_CHECKS:
            raise
    return ret


def log_output_stats(output, info_hist=False, runner_name=None, hist_range=None):
    ret = str_output_stats(output, runner_name)
    G_LOGGER.info(ret)
    with G_LOGGER.indent():
        # For small outputs, show the entire output instead of just a histogram.
        SMALL_OUTPUT_THRESHOLD = 100
        if util.array.size(output) <= SMALL_OUTPUT_THRESHOLD:
            G_LOGGER.log(
                lambda: f"---- Values ----\n{util.indent_block(output)}",
                severity=G_LOGGER.INFO if info_hist else G_LOGGER.VERBOSE,
            )
        G_LOGGER.log(
            lambda: str_histogram(output, hist_range),
            severity=G_LOGGER.INFO if info_hist else G_LOGGER.VERBOSE,
        )


def build_heatmaps(
    arr, min_val, max_val, prefix, save_dir=None, show=None, use_lognorm=None
):
    """
    Display an array as an image or set of images. The last two dimensions are interpreted as
    the height and width and the leading dimensions are flattened and treated as the number
    of images to display.

    Args:
        arr (Union[torch.Tensor, numpy.ndarray]): The input array or tensor.
        min_val (float): The minimum value in the input array
        max_val (float): The maximum value in the input array
        prefix (str): The prefix to use when displaying titles for figures.
        save_dir (Optional[str]): Path to a directory in which to save images of the heatmaps.
        show (Optional[bool]): Whether to display the heatmap.
        use_lognorm (bool): Whether to use LogNorm instead of Normalize when displaying values.
    """
    G_LOGGER.start(f"Building heatmaps for {prefix}. This may take a while...")
    with G_LOGGER.indent():
        MAX_HEIGHT = 1080
        MAX_WIDTH = 1920
        MAX_NUM_ROWS = 14
        MAX_NUM_COLS = 7
        FONT_SIZE = "xx-small"

        shape = util.array.shape(arr)
        if len(shape) < 3:
            arr = util.array.view(
                arr,
                dtype=util.array.dtype(arr),
                shape=([1] * (3 - len(shape))) + list(shape),
            )

        original_shape = util.array.shape(arr)
        arr = util.array.view(
            arr,
            dtype=util.array.dtype(arr),
            shape=(-1, original_shape[-2], original_shape[-1]),
        )

        shape = util.array.shape(arr)
        num_images = shape[0]

        def coord_str_from_img_idx(img_idx):
            coord = []
            for dim in reversed(original_shape[:-2]):
                coord.insert(0, img_idx % dim)
                img_idx //= dim
            return f"({','.join(map(str, coord))},0:{shape[-2]},0:{shape[-1]})"

        # We treat each 2D slice of the array as a separate image.
        # Multiple images may be displayed on a single figure (in a grid) and we may have multiple figures.
        num_rows = min(MAX_HEIGHT // shape[-2], MAX_NUM_ROWS)
        num_cols = min(MAX_WIDTH // shape[-1], MAX_NUM_COLS)

        # Remove any excess images per figure
        if num_images < num_rows * num_cols:
            num_cols = min(num_images, num_cols)
            num_rows = math.ceil(num_images / num_cols)

        num_images_per_figure = num_rows * num_cols
        num_figures = math.ceil(num_images / num_images_per_figure)

        # Populate each image in each figure.
        for fig_idx in range(num_figures):
            fig, axs = plt.subplots(
                num_rows, num_cols, squeeze=False, dpi=200, constrained_layout=True
            )
            base_img_idx = fig_idx * num_images_per_figure

            try:
                # When the error is all the same, we can't use LogNorm.
                if use_lognorm and min_val != max_val:
                    norm = matplotlib.colors.LogNorm(vmin=min_val, vmax=max_val)
                    prefix += " (Log Scale)"
                else:
                    norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)

                fig_title = f"{prefix}: {coord_str_from_img_idx(base_img_idx)} to {coord_str_from_img_idx(min(base_img_idx + num_images_per_figure, num_images) - 1)}"
                fig.suptitle(fig_title, fontsize=FONT_SIZE)

                G_LOGGER.extra_verbose(f"Building heatmaps for {fig_title}")

                images = []
                for row in range(num_rows):
                    for col in range(num_cols):
                        img_idx = base_img_idx + (col + row * num_cols)

                        ax = axs[row, col]
                        ax.set_axis_off()

                        if img_idx < shape[0]:
                            img = arr[img_idx]
                            title = f"{coord_str_from_img_idx(img_idx)}"
                        else:
                            img = np.zeros(shape=(shape[-2:]))
                            title = "Out Of Bounds"
                        ax.set_title(title, fontsize=FONT_SIZE)

                        images.append(
                            ax.imshow(
                                img, cmap="plasma", filternorm=False, resample=False
                            )
                        )

                for im in images:
                    im.set_norm(norm)

                fig.colorbar(images[0], ax=axs, shrink=0.7)

                if save_dir is not None:
                    path = os.path.join(
                        save_dir, f"{util.sanitize_filename(fig_title)}.svg"
                    )
                    util.makedirs(path)
                    G_LOGGER.info(f"Saving '{prefix}' heatmap to: '{path}'")
                    fig.savefig(path)

                if show:
                    plt.show()
            finally:
                plt.close(fig)


def scatter_plot_error_magnitude(
    absdiff,
    reldiff,
    reference_output,
    min_reldiff,
    max_reldiff,
    runner0_name,
    runner1_name,
    out0_name,
    out1_name,
    save_dir=None,
    show=None,
):
    """
    Display a plot of absolute/relative difference against the magnitude of the output.

    Args:
        absdiff (Union[torch.Tensor, numpy.ndarray]): The absolute difference.
        reldiff (Union[torch.Tensor, numpy.ndarray]): The relative difference.
        reference_output (Union[torch.Tensor, numpy.ndarray]): The output to consider as the reference output.
        min_reldiff (float): The minimum relative difference
        max_reldiff (float): The maximum relative difference
        runner0_name (str): The name of the first runner.
        runner1_name (str): The name of the second runner.
        out0_name (str): The name of the output of the first runner.
        out1_name (str): The name of the output of the second runner.
        save_dir (Optional[str]): Path to a directory in which to save images of the plots.
        show (Optional[bool]): Whether to display the error metrics plot.
    """
    G_LOGGER.start(
        f"Building error metrics plot for {out0_name}. This may take a while..."
    )
    with G_LOGGER.indent():
        title = f"Error metrics between output0 and output1\noutput0: {runner0_name:35} | {out0_name}\noutput1: {runner1_name:35} | {out1_name}"
        fname = util.sanitize_filename(f"error_metrics_{out0_name}.png")
        TICK_FONT_SIZE = 6
        TITLE_FONT_SIZE = 7
        NUM_X_TICKS = 20
        NUM_Y_LINEAR_TICKS = 10

        def set_ax_properties(ax):
            ax.tick_params(axis="x", labelrotation=90)
            ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
            ax.grid(linestyle="--")
            ax.xaxis.label.set_fontsize(TITLE_FONT_SIZE)
            ax.yaxis.label.set_fontsize(TITLE_FONT_SIZE)

        def set_linear_ax(ax):
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            ax.set_xticks(np.linspace(0, xticks[-1], NUM_X_TICKS))
            ax.set_yticks(np.linspace(0, yticks[-1], NUM_Y_LINEAR_TICKS))
            set_ax_properties(ax)

        def set_log_ax(ax, min_diff, max_diff):
            ax.set_yscale("log")
            xticks = ax.get_xticks()

            # Add a very small epsilon to prevent division by 0:
            eps = 1e-15
            yrange = np.log10(np.array([min_diff + eps, max_diff + eps]))
            yrange[0] = math.floor(yrange[0])
            yrange[1] = math.ceil(yrange[1])

            ax.set_xticks(np.linspace(0, xticks[-1], NUM_X_TICKS))
            ax.set_yticks(np.power(10, np.arange(yrange[0], yrange[1], 1)))
            set_ax_properties(ax)

        magnitude = util.array.abs(reference_output)
        fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)

        try:
            fig.suptitle(title, fontsize=TITLE_FONT_SIZE)

            axs[0].scatter(magnitude, absdiff, s=1)
            axs[0].set(ylabel="Absolute error")
            set_linear_ax(axs[0])

            axs[1].scatter(magnitude, reldiff, s=1)
            label_suffix = ""
            # When the range of the data is 0, we can't use log scale.
            if min_reldiff != max_reldiff:
                set_log_ax(axs[1], min_reldiff, max_reldiff)
                label_suffix = " (log scale)"
            else:
                set_linear_ax(axs[1])
            axs[1].set(
                xlabel="output1 magnitude", ylabel=f"Relative error{label_suffix}"
            )

            if save_dir is not None:
                path = os.path.join(save_dir, fname)
                util.makedirs(path)
                G_LOGGER.info(f"Saving error metrics plot to: '{path}'")
                fig.savefig(path, dpi=1200, bbox_inches="tight")

            if show:
                plt.show()

        finally:
            plt.close(fig)
