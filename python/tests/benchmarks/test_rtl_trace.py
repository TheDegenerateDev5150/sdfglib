import pytest

from docc.benchmarks import Trace, TraceValidationError, load_schema


def _docc_meta():
    return {
        "sdfg_name": "kernel_0",
        "sdfg_file": "/tmp/sdfg_0.json",
        "arg_capture_path": "",
        "features_file": "",
        "opt_report_file": "",
        "element_id": 10,
        "element_type": "for",
        "loopnest_index": 0,
        "loop_info": {
            "loopnest_index": 0,
            "num_loops": 1,
            "num_maps": 0,
            "num_fors": 1,
            "num_whiles": 0,
            "max_depth": 1,
            "is_perfectly_nested": True,
            "is_perfectly_parallel": False,
            "is_elementwise": False,
            "has_side_effects": False,
        },
    }


def _args(metrics):
    return {
        "function": "main",
        "module": "kernel.c",
        "source_ranges": [
            {
                "file": "kernel.c",
                "from": {"line": 1, "col": 1},
                "to": {"line": 9, "col": 2},
            }
        ],
        "docc": _docc_meta(),
        "target_type": "sequential",
        "metrics": metrics,
    }


def _per_invocation_trace():
    return {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "region,daisy",
                "name": "main [L1-9]",
                "pid": 1,
                "tid": 1,
                "ts": 100,
                "dur": 250,
                "args": _args({"perf::CYCLES": 1000, "perf::BRANCHES": 20}),
            }
        ]
    }


def _aggregated_trace():
    return {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "aggregated_region,daisy",
                "name": "main [L1-9]",
                "pid": 1,
                "tid": 1,
                "ts": 100,
                "dur": 2500,
                "args": _args(
                    {
                        "perf::CYCLES": {
                            "mean": 1000.0,
                            "variance": 5.0,
                            "count": 10,
                            "min": 990,
                            "max": 1010,
                        },
                        "static:::flop": {
                            "mean": 512.0,
                            "variance": 0.0,
                            "count": 10,
                            "min": 512,
                            "max": 512,
                        },
                        "static:::ratio": {
                            "mean": None,
                            "variance": None,
                            "count": 1,
                            "min": None,
                            "max": None,
                        },
                        "runtime": {
                            "mean": 250.0,
                            "variance": 1.5,
                            "count": 10,
                            "min": 240.0,
                            "max": 260.0,
                        },
                    }
                ),
            }
        ]
    }


def test_load_schema_available():
    schema = load_schema()
    assert schema["$id"].endswith("daisy_trace.schema.json")


def test_per_invocation_trace():
    trace = Trace.from_dict(_per_invocation_trace())
    assert len(trace) == 1
    assert not trace.is_aggregated

    region = trace[0]
    assert region.name == "main [L1-9]"
    assert region.target_type == "sequential"
    assert region.has_element
    assert region.element_type == "for"
    assert region.loop_info.num_fors == 1
    assert region.source_ranges[0].begin.line == 1

    assert region.counters == {"perf::CYCLES": 1000, "perf::BRANCHES": 20}
    assert region.metric("perf::CYCLES") == 1000
    assert region.runtime_mean_us == 250
    assert region.counter_stats == {}


def test_aggregated_trace():
    trace = Trace.from_dict(_aggregated_trace())
    assert trace.is_aggregated

    region = trace[0]
    assert region.runtime is not None
    assert region.runtime_mean_us == 250.0
    assert region.counter_stats["perf::CYCLES"].mean == 1000.0
    # static:::-prefixed metrics are exposed without the prefix and excluded from counters
    assert set(region.static_metrics) == {"flop", "ratio"}
    assert "perf::CYCLES" in region.counter_stats
    assert "flop" not in region.counter_stats
    assert region.static_metrics["ratio"].mean is None
    assert region.metric("flop") == 512.0


def test_queries():
    trace = Trace.from_dict(_aggregated_trace())
    assert trace.filter(target_type="sequential")
    assert not trace.filter(target_type="CUDA")
    assert trace.by_element_id(10)
    assert trace.hottest(1)[0].name == "main [L1-9]"
    assert trace.total_runtime_us() == 250.0


def test_invalid_trace_raises():
    with pytest.raises(TraceValidationError) as exc:
        Trace.from_dict({"traceEvents": [{"ph": "B"}]})
    assert exc.value.errors


def test_validation_can_be_skipped():
    # Structurally-invalid trace loads when validation is disabled.
    trace = Trace.from_dict({"traceEvents": []}, validate_schema=False)
    assert len(trace) == 0
