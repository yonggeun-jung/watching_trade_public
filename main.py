import subprocess

STEPS = [
    "watching_trade/src_data/run_data.py",     # If you don't need data processing, you can skip this step
    "watching_trade/src_models/run_models.py",
    "watching_trade/src_russia/run_russia.py",
    "watching_trade/src_simulation/run_simulation.py",
    "watching_trade/scr_mis/run_mis.py"]


if __name__ == "__main__":
    print("\nStarting execution...\n")
    
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
            print("Aborted.")
            raise SystemExit(returncode)

    print("\nCompleted.\n")