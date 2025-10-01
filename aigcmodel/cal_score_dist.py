import os
import csv
import numpy as np

# Directory containing the files
data_dir = './data'  # Change this to your target directory path
list_perf = list()
csv_header = ["filename"]
len_dist = 2
bin_edges = np.linspace(0.9, 1.8, len_dist+1)

# Traverse files that match the pattern
for filename in os.listdir(data_dir):
    # Score bins
    hist_total = np.zeros(len(bin_edges) - 1, dtype=int)
    _list_perf = list()
    file_path = os.path.join(data_dir, filename)
    if os.path.isfile(file_path) and filename.startswith("eval_#200_score") and filename.endswith(".csv"):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    try:
                        score = float(row[-1])
                        # Count into bins
                        hist, _ = np.histogram([score], bins=bin_edges)
                        hist_total += hist
                    except ValueError:
                        continue  # skip malformed rows

            print(f"Aggregated Score Distribution of {filename}:")
            _list_perf.append(filename)
            for i in range(len(hist_total)):
                perf_model = round((hist_total[i] / 200), 4)
                _list_perf.append(perf_model)
                print(f"Range {bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}: {perf_model}")

            list_perf.append(_list_perf)
            print(f"Check the sum of the number of score is equal to 200: {sum(hist_total)}")

for i in range(len(bin_edges) - 1):
    csv_header.append(f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}")

with open(os.path.join(data_dir, f"model_perf_dist_{len_dist}.csv"), mode="w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)  # Write header first
    writer.writerows(list_perf)   # Write all data rows
