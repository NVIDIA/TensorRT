import functools

from polygraphy import mod, util, config
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")


def zero_on_empty(func):
    @functools.wraps(func)
    def wrapped(buffer):
        if util.is_empty_shape(buffer.shape):
            return 0
        return func(buffer)

    return wrapped


@zero_on_empty
def compute_max(buffer):
    return np.amax(buffer)


# Returns index of max value
@zero_on_empty
def compute_argmax(buffer):
    return np.unravel_index(np.argmax(buffer), buffer.shape)


@zero_on_empty
def compute_min(buffer):
    return np.amin(buffer)


# Returns index of min value
@zero_on_empty
def compute_argmin(buffer):
    return np.unravel_index(np.argmin(buffer), buffer.shape)


@zero_on_empty
def compute_mean(buffer):
    return np.mean(buffer)


@zero_on_empty
def compute_stddev(buffer):
    return np.std(buffer)


@zero_on_empty
def compute_variance(buffer):
    return np.var(buffer)


@zero_on_empty
def compute_median(buffer):
    return np.median(buffer)


@zero_on_empty
def compute_average_magnitude(buffer):
    return np.mean(np.abs(buffer))


def str_histogram(output, hist_range=None):
    if np.issubdtype(output.dtype, np.bool_):
        return ""

    try:
        try:
            hist, bin_edges = np.histogram(output, range=hist_range)
        except ValueError as err:
            G_LOGGER.verbose("Could not generate histogram. Note: Error was: {:}".format(err))
            return ""

        max_num_elems = compute_max(hist)
        if not max_num_elems:  # Empty tensor
            return

        bin_edges = ["{:.3g}".format(bin) for bin in bin_edges]
        max_start_bin_width = max(len(bin) for bin in bin_edges)
        max_end_bin_width = max(len(bin) for bin in bin_edges[1:])

        MAX_WIDTH = 40
        ret = "---- Histogram ----\n"
        ret += "{:{width}}|  Num Elems | Visualization\n".format(
            "Bin Range", width=max_start_bin_width + max_end_bin_width + 5
        )
        for num, bin_start, bin_end in zip(hist, bin_edges, bin_edges[1:]):
            bar = "#" * int(MAX_WIDTH * float(num) / float(max_num_elems))
            ret += "({:<{max_start_bin_width}}, {:<{max_end_bin_width}}) | {:10} | {:}\n".format(
                bin_start,
                bin_end,
                num,
                bar,
                max_start_bin_width=max_start_bin_width,
                max_end_bin_width=max_end_bin_width,
            )
        return ret
    except Exception as err:
        G_LOGGER.verbose("Could not generate histogram.\nNote: Error was: {:}".format(err))
        if config.INTERNAL_CORRECTNESS_CHECKS:
            raise
        return ""


def str_output_stats(output, runner_name=None):
    ret = ""
    if runner_name:
        ret += "{:} | Stats: ".format(runner_name)

    try:
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            ret += "mean={:.5g}, std-dev={:.5g}, var={:.5g}, median={:.5g}, min={:.5g} at {:}, max={:.5g} at {:}, avg-magnitude={:.5g}\n".format(
                compute_mean(output),
                compute_stddev(output),
                compute_variance(output),
                compute_median(output),
                compute_min(output),
                compute_argmin(output),
                compute_max(output),
                compute_argmax(output),
                compute_average_magnitude(output),
            )
    except Exception as err:
        G_LOGGER.verbose("Could not generate statistics.\nNote: Error was: {:}".format(err))
        ret += "<Error while computing statistics>"
        if config.INTERNAL_CORRECTNESS_CHECKS:
            raise
    return ret


def log_output_stats(output, info_hist=False, runner_name=None, hist_range=None):
    ret = str_output_stats(output, runner_name)
    G_LOGGER.info(ret)
    with G_LOGGER.indent():
        # Show histogram on failures.
        G_LOGGER.log(
            lambda: str_histogram(output, hist_range), severity=G_LOGGER.INFO if info_hist else G_LOGGER.VERBOSE
        )
