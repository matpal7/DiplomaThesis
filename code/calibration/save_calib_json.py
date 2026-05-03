import json
import os
import numpy as np


def _convert_for_json(obj):
    """
    Recursively convert numpy types to JSON-serializable Python types.
    """

    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    return obj


def save_dict_json(
    calib_dict,
    out_folder,
    file_name_npy="calib_data.npy",
    file_name_json="calib_data.json",
):
    """
    Save calibration dictionary both as:
      - NumPy .npy
      - human-readable JSON
    """

    os.makedirs(out_folder, exist_ok=True)

    # -----------------------------
    # Save NPY
    # -----------------------------
    npy_path = os.path.join(out_folder, file_name_npy)
    np.save(npy_path, calib_dict)

    print("Wrote calib data to:", npy_path)

    # -----------------------------
    # Save JSON
    # -----------------------------
    json_path = os.path.join(out_folder, file_name_json)

    calib_json = _convert_for_json(calib_dict)

    with open(json_path, "w") as f:
        json.dump(calib_json, f, indent=4)

    print("Wrote calib JSON to:", json_path)
