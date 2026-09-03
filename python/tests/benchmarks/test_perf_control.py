from io import StringIO

from docc.benchmarks import PerfControl


MARKER = "#### DAISYTUNER Marker ####"


def test_automatic_mode_is_noop(tmp_path):
    output = tmp_path / "runtime.csv"
    perf = PerfControl.from_env(
        {
            "DAISY_CI_MEASUREMENT_MODE": "automatic",
            "DAISY_CI_RUNTIME_OUTPUT": str(output),
        },
        verbose=False,
    )

    assert perf.mode == "automatic"
    assert not perf.enabled

    with perf.measure():
        # No-op
        perf.record_metric("energy-pkg", 1.0, "Joules")

    assert not output.exists()


def test_control_mode_sends_enable_disable():
    ctl = StringIO()
    ack = StringIO("ack\nack\n")
    perf = PerfControl(ctl, ack, mode="control", verbose=False)

    with perf.measure():
        assert perf.active

    assert not perf.active
    assert ctl.getvalue().splitlines() == ["enable", "disable"]


def test_manual_mode_measure_does_not_write_implicit_duration(tmp_path):
    output = tmp_path / "runtime.csv"
    perf = PerfControl.from_env(
        {
            "DAISY_CI_MEASUREMENT_MODE": "manual",
            "DAISY_CI_RUNTIME_OUTPUT": str(output),
            "DAISY_CI_MEASUREMENTS": "3",
        },
        verbose=False,
    )

    assert perf.mode == "manual"
    assert perf.enabled
    assert perf.measurements == 3

    with perf.measure():
        pass

    assert not output.exists()


def test_manual_mode_measure_callback_writes_provided_metrics(tmp_path):
    output = tmp_path / "runtime.csv"
    perf = PerfControl.from_env(
        {
            "DAISY_CI_MEASUREMENT_MODE": "manual",
            "DAISY_CI_RUNTIME_OUTPUT": str(output),
        },
        verbose=False,
    )

    result = perf.measure_callback(
        lambda: {
            "duration_time": (125, "ms"),
            "energy-pkg": (0.75, "Joules"),
            "custom_score": 9,
        }
    )

    assert result == {
        "duration_time": (125, "ms"),
        "energy-pkg": (0.75, "Joules"),
        "custom_score": 9,
    }
    assert output.read_text(encoding="utf-8").splitlines() == [
        "duration_time,ms,125",
        "energy-pkg,Joules,0.75",
        "custom_score,,9",
        MARKER,
    ]


def test_manual_mode_write_measurement_accepts_extra_metrics(tmp_path):
    output = tmp_path / "runtime.csv"
    perf = PerfControl.from_env(
        {
            "DAISY_CI_MEASUREMENT_MODE": "manual",
            "DAISY_CI_RUNTIME_OUTPUT": str(output),
        },
        verbose=False,
    )

    perf.write_measurement(
        duration_time=125,
        duration_unit="ms",
        energy=0.75,
        metrics={"cache_misses": (42, "count"), "custom_score": 9},
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "duration_time,ms,125",
        "energy-pkg,Joules,0.75",
        "cache_misses,count,42",
        "custom_score,,9",
        MARKER,
    ]


def test_manual_mode_write_measurements_separates_each_block(tmp_path):
    output = tmp_path / "runtime.csv"
    perf = PerfControl.from_env(
        {
            "DAISY_CI_MEASUREMENT_MODE": "manual",
            "DAISY_CI_RUNTIME_OUTPUT": str(output),
        },
        verbose=False,
    )

    perf.write_measurements(
        [
            {"duration_time": (100, "ms"), "energy-pkg": (0.10, "Joules")},
            {"duration_time": (110, "ms"), "energy-pkg": (0.12, "Joules")},
            {"duration_time": (105, "ms"), "energy-pkg": (0.11, "Joules")},
        ]
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "duration_time,ms,100",
        "energy-pkg,Joules,0.1",
        MARKER,
        "duration_time,ms,110",
        "energy-pkg,Joules,0.12",
        MARKER,
        "duration_time,ms,105",
        "energy-pkg,Joules,0.11",
        MARKER,
    ]
