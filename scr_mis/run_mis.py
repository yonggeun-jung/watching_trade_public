import subprocess

STEPS = [
    "watching_trade/scr_mis/01_sar_example.py",
    "watching_trade/scr_mis/02_summary_stat.py",
    "watching_trade/scr_mis/03_nowcasting_plots_val.py",
    "watching_trade/scr_mis/04_nowcasting_plots_wgt.py",
    "watching_trade/scr_mis/05_LOO_plots_val.py",
    "watching_trade/scr_mis/06_LOO_plots_wgt.py",
    "watching_trade/scr_mis/07_russia_plot.py",
    "watching_trade/scr_mis/08_flow_chart.py"]


if __name__ == "__main__":
    print("\nStarting...\n")
    
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

# CHECKED.