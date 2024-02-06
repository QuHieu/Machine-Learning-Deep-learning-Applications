import numpy as np
import csv

csv_file_path = 'output.csv'

with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for item in y_test_d_predictions:
        csvwriter.writerow([item])

print(f"Array has been written to {csv_file_path}")