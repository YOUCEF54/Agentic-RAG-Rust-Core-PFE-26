"""Calibrate hardware parameters for embedding/indexing workloads."""

from get_hardware_config import run_hardware_calibration


if __name__ == "__main__":
    result = run_hardware_calibration(save_config=True, quick_mode=True, max_runtime_seconds=25.0)
    print(result)

