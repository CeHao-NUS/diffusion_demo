import subprocess
from multiprocessing import Pool

# def run_script(args):
#     folder_name, loss_weight = args
#     subprocess.run(["python", "eval.py", "--folder_name=" + folder_name, "--loss_weight=" + str(loss_weight)])

# if __name__ == "__main__":
#     params = [
#         ("cubic_w1", 1),
#         ("cubic_w2", 2),
#         ("cubic_w4", 4),
#         ("cubic_w8", 8),
#         ("cubic_w16", 16),
#         ("cubic_w32", 32),
#         ("cubic_w64", 64),
#     ]

#     with Pool() as pool:
#         pool.map(run_script, params)


def run_script(args):
    folder_name, threshold = args
    subprocess.run(["python", "eval.py", "--folder_name=" + folder_name, "--threshold=" + str(threshold)])

if __name__ == "__main__":
    params = [
        ("inequ_-2", 0.01),
        ("inequ_-3", 0.001),
        ("inequ_-4", 0.0001),
        ("inequ_-5", 0.00001),
    ]

    with Pool() as pool:
        pool.map(run_script, params)


