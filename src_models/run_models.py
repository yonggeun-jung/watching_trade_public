import subprocess

STEPS = [
    "watching_trade/src_models/01_ols.py",
    "watching_trade/src_models/02_xgb_wgt_ports.py",
    "watching_trade/src_models/03_xgb_wgt_NoPorts.py",
    "watching_trade/src_models/04_xgb_val_ports.py",
    "watching_trade/src_models/05_xgb_val_NoPorts.py",
    "watching_trade/src_models/06_xgb_wgt_ports_LOO.py",
    "watching_trade/src_models/07_xgb_wgt_NoPorts_LOO.py",
    "watching_trade/src_models/08_xgb_val_ports_LOO.py",
    "watching_trade/src_models/09_xgb_val_NoPorts_LOO.py",
    "watching_trade/src_models/10_xgb_val_NoPort_placebo.py"]


if __name__ == "__main__":
    print("\nStarting model execution...\n")
    
    for step in STEPS:

        process = subprocess.Popen(
            ["python3", step],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for line in process.stdout:
            print(line, end="")

        for line in process.stderr:
            print(line, end="")

        returncode = process.wait()
        if returncode == 0:
            print(f"--- Finished {step} ---")
        else:
            print(f"*** ERROR in {step}, return code {returncode} ***")
            print("Model execution aborted.")
            raise SystemExit(returncode)

    print("\nModel execution completed.\n")

# CHECKED