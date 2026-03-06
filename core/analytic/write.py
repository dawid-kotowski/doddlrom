import os
import csv


def append_error_decomp_csv(
    csv_path: str,
    example_name: str,
    model_name: str,
    metrics: dict,
    P,
    test_samples: int,
    dedup: bool = True
) -> bool:
    """
    Appends a single row with error-decomposition metrics.
    Returns True if a row was written, False if skipped (due to dedup).
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if model_name.lower() == "pod-dl-rom":
        dim_field = "N"
        dim_val = int(P.N)
        specific = {
            "N": dim_val,
            "E_POD": f"{metrics['E_POD']:.6e}",
            "E_POD_inf": f"{metrics['E_POD_inf']:.6e}",
        }
        fieldnames = [
            "example","model","test_samples","N","m","M",
            "E_R","E_S","E_POD","E_POD_inf","E_NN",
            "upper_bound","lower_bound"
        ]
    else:  # "dod-dl-rom"
        dim_field = "N_prime"
        dim_val = int(P.N_prime)
        specific = {
            "N_prime": dim_val,
            "E_DOD": f"{metrics['E_DOD']:.6e}",
            "E_DOD_inf": f"{metrics['E_DOD_inf']:.6e}",
        }
        fieldnames = [
            "example","model","test_samples","N_prime","m","M",
            "E_R","E_S","E_DOD","E_DOD_inf","E_NN",
            "upper_bound","lower_bound"
        ]

    common = {
        "example": example_name,
        "model": model_name,
        "test_samples": test_samples,
        "m": f"{metrics['m']:.6e}",
        "M": f"{metrics['M']:.6e}",
        "E_R": f"{metrics['E_R']:.6e}",
        "E_S": f"{metrics['E_S']:.6e}",
        "E_NN": f"{metrics['E_NN']:.6e}",
        "upper_bound": f"{metrics['upper_bound']:.6e}",
        "lower_bound": f"{metrics['lower_bound']:.6e}",
    }
    row = {**common, **specific}

    exists = os.path.exists(csv_path)
    if dedup and exists:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if (
                    r.get("example") == example_name
                    and r.get("model") == model_name
                    and r.get(dim_field) == str(dim_val)
                ):
                    return False 

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    return True
