import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, norm
from itertools import product
from itertools import combinations

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, norm

# Load the normalization file once
norm_mats = scipy.io.loadmat('E:/courses/IISC/all_MATfiles/all_MATfiles/all-20240626T062008Z-001/all/UPU/UPU.mat')
UL_norm = []
VEL_norm = []

for annotation in norm_mats['annotation'][0]:
    _, _, _, UL, _, _, VEL, _ = annotation
    UL = np.squeeze(UL)
    VEL = np.squeeze(VEL)
    UL_norm.append(UL)
    VEL_norm.append(VEL)

UL_norm_array = np.array(UL_norm)
VEL_norm_array = np.array(VEL_norm)
norm_distances = np.sqrt(np.sum((UL_norm_array - VEL_norm_array)**2, axis=1))

# Process the original distances and normalize them
all_distances_norm = []
sounds = ["UPU-sub01", "UPU-sub02", "UPU-sub03", "UPU-sub04", "UPU-sub05", "UPU-sub06", "UPU-sub07", "UPU-sub08", "UPU-sub09", "UPU-sub010",
          "UPU-sub011", "UPU-sub012", "UPU-sub013", "UPU-sub014", "UPU-sub015", "UPU-sub016", "UPU-sub017", "UPU-sub018", "UPU-sub019", "UPU-sub020",
          "UPU-sub021", "UPU-sub022", "UPU-sub023", "UPU-sub024", "UPU-sub025", "UPU-sub026", "UPU-sub027", "UPU-sub028", "UPU-sub029", "UPU-sub030",
          "UPU-sub031", "UPU-sub032", "UPU-sub033", "UPU-sub034", "UPU-sub035", "UPU-sub036", "UPU-sub037", "UPU-sub038", "UPU-sub039", "UPU-sub040",
          "UPU-sub042", "UPU-sub043", "UPU-sub044", "UPU-sub045", "UPU-sub046", "UPU-sub047", "UPU-sub048", "UPU-sub049", "UPU-sub050",
          "UPU-sub051", "UPU-sub052", "UPU-sub053", "UPU-sub054", "UPU-sub055", "UPU-sub056", "UPU-sub057", "UPU-sub058", "UPU-sub059", "UPU-sub060",
          "UPU-sub061", "UPU-sub062", "UPU-sub063", "UPU-sub064", "UPU-sub065", "UPU-sub066", "UPU-sub067", "UPU-sub068", "UPU-sub069", "UPU-sub070",
          "UPU-sub071", "UPU-sub072", "UPU-sub073", "UPU-sub074", "UPU-sub075"]

for i, sound in enumerate(sounds):
    mats = scipy.io.loadmat('E:/courses/IISC_working/P/P/UPU/' + sound + '.mat')
    
    UL_list = []
    LL_list = []

    for annotation in mats['annotation'][0]:
        _, _, _, UL, LL, _, _, _, _, _ = annotation
        UL = np.squeeze(UL)
        LL = np.squeeze(LL)
        UL_list.append(UL)
        LL_list.append(LL)
    
    UL_array = np.array(UL_list)
    LL_array = np.array(LL_list)
    
    distances = np.sqrt(np.sum((UL_array - LL_array)**2, axis=1))
    norm_distance = norm_distances[i] if i < len(norm_distances) else 1  # to handle index errors if any
    normalized_distances = distances / norm_distance
    all_distances_norm.append(normalized_distances)
    print(f"Processed {sound}")

# Convert list of arrays to a 2D NumPy array
all_distances_norm_array = np.array(all_distances_norm)

# Calculate the mean distances across all sounds
mean_distances_norm = np.mean(all_distances_norm_array, axis=0)
# Calculate the variance and standard deviation across all sounds
variance_distances_norm = np.var(all_distances_norm_array, axis=0)
std_distances_norm = np.std(all_distances_norm_array, axis=0)

print("Mean normalized distances across all sounds:")
print(mean_distances_norm)

print("\nVariance of normalized distances across all sounds:")
print(variance_distances_norm)

print("\nStandard deviation of normalized distances across all sounds:")
print(std_distances_norm)

# Define the specified indices
indices_mapping = [3, 4, 2, 5, 1, 6, 0]

# Create a new array for the mean distances with the specified indices
new_mean_array_norm = np.empty(len(indices_mapping))
new_variance_array_norm = np.empty(len(indices_mapping))
new_std_array_norm = np.empty(len(indices_mapping))

for i, new_index in enumerate(indices_mapping):
    new_mean_array_norm[new_index] = mean_distances_norm[i]
    new_variance_array_norm[new_index] = variance_distances_norm[i]
    new_std_array_norm[new_index] = std_distances_norm[i]

print("New array with mean normalized distances in specified indices:")
print(new_mean_array_norm)

print("\nNew array with variance of normalized distances in specified indices:")
print(new_variance_array_norm)

print("\nNew array with standard deviation of normalized distances in specified indices:")
print(new_std_array_norm)

# Define x-axis points
x_points = np.array([-3, -2, -1, 0, 1, 2, 3])

# Calculate Wilson score intervals
def wilson_score_interval(mean, std, n, confidence_level=0.95):
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    factor = z * (std / np.sqrt(n))
    lower_bound = mean - factor
    upper_bound = mean + factor
    return lower_bound, upper_bound

n = len(all_distances_norm_array)
wilson_intervals = [wilson_score_interval(new_mean_array_norm[i], new_std_array_norm[i], n) for i in range(len(new_mean_array_norm))]
lower_bounds = [interval[0] for interval in wilson_intervals]
upper_bounds = [interval[1] for interval in wilson_intervals]

# Print Wilson intervals
print("\nWilson Score for each point: ")
for i, (lb, ub) in enumerate(zip(lower_bounds, upper_bounds)):
    print(f"Point {x_points[i]}: Lower bound = {lb}, Upper bound = {ub}")

# Plotting
plt.figure(figsize=(12, 8))
# Plot each individual distance curve with the new indices
for distances in all_distances_norm:
    new_distances = np.empty(len(indices_mapping))
    for i, new_index in enumerate(indices_mapping):
        new_distances[new_index] = distances[i]
    plt.plot(x_points, new_distances, marker='o', linestyle='-', color='skyblue', alpha=0.5)

# Plot the mean distances in a different color
plt.plot(x_points, new_mean_array_norm, marker='o', linestyle='-', color='r', linewidth=2, label='Mean distance')

# Plot the Wilson score intervals
# plt.fill_between(x_points, lower_bounds, upper_bounds, color='orange', alpha=0.3, label='Wilson score interval')

plt.xlabel('Point Index')
plt.ylabel('Normalized Euclidean Distance')
plt.title('Normalized Euclidean Distances Between Corresponding UL and LL Points Across All Sounds with Wilson Intervals')
plt.grid(True)
plt.legend()
plt.show()

# Plotting standard deviation as bar graph
plt.figure(figsize=(12, 8))
plt.bar(x_points, new_std_array_norm, color='blue', alpha=0.5)

plt.xlabel('Point Index')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of Normalized Euclidean Distances Between Corresponding UL and LL Points Across All Sounds')
plt.grid(True)
plt.show()

# Perform Wilcoxon signed-rank test on the normalized distances
for i, distances_norm in enumerate(all_distances_norm):
    wilcoxon_result = wilcoxon(distances_norm, new_mean_array_norm)
    print(f"Wilcoxon test result for {sounds[i]}: {wilcoxon_result}")

# Perform paired t-test on the normalized distances
for i, distances_norm in enumerate(all_distances_norm):
    ttest_result = ttest_rel(distances_norm, new_mean_array_norm)
    print(f"T-test result for {sounds[i]}: {ttest_result}")

# Threshold for determining outliers (common thresholds: 0.05, 0.01)
p_value_threshold = 0.05

# List to hold outlier information
outliers = []

# Perform paired t-test on the normalized distances
p_values = []
for i, distances_norm in enumerate(all_distances_norm):
    ttest_result = ttest_rel(distances_norm, new_mean_array_norm)
    p_value = ttest_result.pvalue
    p_values.append(p_value)
    
    # Check if this p-value indicates an outlier
    if p_value < p_value_threshold:
        outliers.append(i)
    
    print(f"T-test result for {sounds[i]}: p-value = {p_value}")

# Convert p-values list to numpy array for easier plotting
p_values_array = np.array(p_values)

# Plot p-values
plt.figure(figsize=(12, 8))
plt.plot(np.arange(len(sounds)), p_values_array, marker='o', linestyle='-', color='g', linewidth=2, label='p-values')

# Highlight outliers
plt.scatter(outliers, p_values_array[outliers], color='red', label='Outliers', zorder=5)

plt.axhline(y=p_value_threshold, color='r', linestyle='--', label=f'p-value threshold = {p_value_threshold}')
plt.xlabel('Sound Index')
plt.ylabel('p-value')
plt.title('Paired T-test p-values for Normalized Distances Across Sounds for UPU')
plt.grid(True)
plt.legend()
plt.show()

print("\nOutliers based on p-value threshold:")
for outlier_index in outliers:
    print(f"Sound: {sounds[outlier_index]}, Index: {outlier_index}, p-value: {p_values[outlier_index]}")

# For checking similarity

# Indices of points before and after the consonant frame
before_indices = [0, 1, 2]  # Corresponding to points -3, -2, -1
after_indices = [6, 5, 4]    # Corresponding to points 3, 2, 1

# # Perform paired t-tests for the specified pairs
# for before_idx, after_idx in product(before_indices, after_indices):
#     # Extract data points before and after the consonant frame
#     before_data = all_distances_norm_array[:, before_idx]
#     after_data = all_distances_norm_array[:, after_idx]
    
#     # Perform the paired t-test
#     ttest_result = ttest_rel(before_data, after_data)
    
#     print(f"T-test result between points {x_points[before_idx]} & {x_points[after_idx]}: p-value = {ttest_result.pvalue}, t-statistic = {ttest_result.statistic}")
    
    
# Additionally, perform t-tests for combinations within the 'before' and 'after' indices themselves
for idx1, idx2 in combinations(before_indices + after_indices, 2):
    data1 = all_distances_norm_array[:, idx1]
    data2 = all_distances_norm_array[:, idx2]
    
    ttest_result = ttest_rel(data1, data2)
    print(f"T-test result between points {x_points[idx1]} & {x_points[idx2]}: p-value = {ttest_result.pvalue}")