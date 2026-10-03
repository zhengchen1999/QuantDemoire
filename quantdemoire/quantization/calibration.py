"""Two-pass activation and weight calibration."""

from quantdemoire.data import create_dataset

from .pipeline import (
    compute_smooth_scales,
    finalize_activation_ranges,
    quantize_model_weights,
    set_calibration_mode,
)


def _get_attr(source, name, default=None):
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _calibration_loader(args, data_path, batch_size, sample_num, mode):
    return create_dataset(
        args,
        data_path=data_path,
        mode=mode,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        sample_num=sample_num,
    )


def run_calibration_pass(
    args,
    model,
    model_fn_test,
    compute_metrics,
    data_path,
    batch_size,
    sample_num,
    pass_mode,
    loader_mode="calibration",
):
    loader = _calibration_loader(
        args=args,
        data_path=data_path,
        batch_size=batch_size,
        sample_num=sample_num,
        mode=loader_mode,
    )
    set_calibration_mode(model, pass_mode)
    model.eval()
    original_eval_metric = args.EVALUATION_METRIC
    args.EVALUATION_METRIC = False
    try:
        for data in loader:
            model_fn_test(args, data, model, save_path=None, compute_metrics=compute_metrics)
    finally:
        args.EVALUATION_METRIC = original_eval_metric
        set_calibration_mode(model, 0)


def run_two_pass_calibration(args, model, model_fn_test, compute_metrics=None):
    cali_dataset = _get_attr(args, "CALI_DATASET", _get_attr(args, "TRAIN_DATASET"))
    cali_num = int(_get_attr(args, "CALI_NUM", 32))
    cali_batch_size = int(_get_attr(args, "CALI_BATCH_SIZE", 1))
    loader_mode = "train" if _get_attr(args, "CALI_USE_TRAIN_CROPS", False) else "calibration"

    run_calibration_pass(
        args=args,
        model=model,
        model_fn_test=model_fn_test,
        compute_metrics=compute_metrics,
        data_path=cali_dataset,
        batch_size=cali_batch_size,
        sample_num=cali_num,
        pass_mode=2,
        loader_mode=loader_mode,
    )
    compute_smooth_scales(model)
    run_calibration_pass(
        args=args,
        model=model,
        model_fn_test=model_fn_test,
        compute_metrics=compute_metrics,
        data_path=cali_dataset,
        batch_size=cali_batch_size,
        sample_num=cali_num,
        pass_mode=1,
        loader_mode=loader_mode,
    )
    finalize_activation_ranges(model)
    quantize_model_weights(model)
