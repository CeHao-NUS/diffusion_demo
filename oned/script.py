import subprocess
from multiprocessing import Pool

def run_script(args):
    folder_name, loss_weight = args
    subprocess.run(["python", "eval.py", "--folder_name=" + folder_name, "--loss_weight=" + str(loss_weight)])

if __name__ == "__main__":
    params = [
        ("cubic_w1", 1),
        ("cubic_w2", 2),
        ("cubic_w4", 4),
        ("cubic_w8", 8),
        ("cubic_w16", 16),
        ("cubic_w32", 32),
        ("cubic_w64", 64),
    ]

    with Pool() as pool:
        pool.map(run_script, params)


