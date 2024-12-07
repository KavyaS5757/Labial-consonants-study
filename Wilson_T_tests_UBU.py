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
sounds = ["UBU-sub01", "UBU-sub02", "UBU-sub03", "UBU-sub04", "UBU-sub05", "UBU-sub06", "UBU-sub07", "UBU-sub08", "UBU-sub09", "UBU-sub010",
          "UBU-sub011", "UBU-sub012", "UBU-sub013", "UBU-sub014", "UBU-sub015", "UBU-sub016", "UBU-sub017", "UBU-sub018", "UBU-sub019", "UBU-sub020",
          "UBU-sub021", "UBU-sub022", "UBU-sub023", "UBU-sub024", "UBU-sub025", "UBU-sub026", "UBU-sub027", "UBU-sub028", "UBU-sub029", "UBU-sub030",
          "UBU-sub031", "UBU-sub032", "UBU-sub033", "UBU-sub034", "UBU-sub035", "UBU-sub036", "UBU-sub037", "UBU-sub038", "UBU-sub039", "UBU-sub040",
          "UBU-sub042", "UBU-sub043", "UBU-sub044", "UBU-sub045", "UBU-sub046", "UBU-sub047", "UBU-sub048", "UBU-sub049", "UBU-sub050",
          "UBU-sub051", "UBU-sub052", "UBU-sub053", "UBU-sub054", "UBU-sub055", "UBU-sub056", "UBU-sub057", "UBU-sub058", "UBU-sub059", "UBU-sub060",
          "UBU-sub061", "UBU-sub062", "UBU-sub063", "UBU-sub064", "UBU-sub065", "UBU-sub066", "UBU-sub067", "UBU-sub068", "UBU-sub069", "UBU-sub070",
          "UBU-sub071", "UBU-sub072", "UBU-sub073", "UBU-sub074", "UBU-sub075"]

for i, sound in enumerate(sounds):
    mats = scipy.io.loadmat('E:/courses/IISC_working/UBU/UBU/' + sound + '.mat')
    
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

# Plot the mean distances in a different color
plt.plot(x_points, new_mean_array_norm, marker='o', linestyle='-', color='r', linewidth=2, label='Mean normalized distance')

# Plot the Wilson score intervals
plt.fill_between(x_points, lower_bounds, upper_bounds, color='orange', alpha=0.3, label='Wilson score interval')

plt.xlabel('Point Index')
plt.ylabel('Normalized Euclidean Distance')
plt.title('Normalized Euclidean Distances Between Corresponding UL and LL Points Across All Sounds with Wilson Intervals(UBU)')
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

print(outliers)
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
plt.title('Paired T-test p-values for Normalized Distances Across Sounds')
plt.grid(True)
plt.legend()
plt.show()

print("\nOutliers based on p-value threshold:")
for outlier_index in outliers:
    print(f"Sound: {sounds[outlier_index]}, p-value: {p_values[outlier_index]}")


import matplotlib.pyplot as plt
import numpy as np

# Assuming p-values array is already defined
p_values = np.array(p_values)

# Define the outlier threshold
outlier_threshold = 0.05
outliers = p_values[p_values < outlier_threshold]

# Create the histogram
plt.figure(figsize=(12, 8))
plt.hist(p_values, bins=20, color='skyblue', alpha=0.7, edgecolor='black', density=True)

# Overlay the outliers on the histogram
plt.scatter(outliers, np.zeros_like(outliers), color='red', zorder=5, label='Outliers (p < 0.05)', marker='x')

# Set custom x and y limits to start from (-1, -1)
plt.xlim(0, 1)  # X-axis from -1 to 1 (even though p-values are non-negative)
plt.ylim(-0.5, plt.ylim()[1])  # Y-axis from -1 to the max density

plt.title('Histogram of p-values with Outliers Highlighted')
plt.xlabel('p-value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()
plt.show()

# 2. Boxplot of p-values
plt.figure(figsize=(12, 8))
plt.boxplot(p_values, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen', color='black'))
plt.title('Boxplot of p-values')
plt.xlabel('p-value')
plt.grid(True)
plt.show()

# 3. Stick to Standard Thresholds: Plot p-values and highlight significance
p_value_threshold = 0.05
significant_indices = np.where(p_values < p_value_threshold)[0]

plt.figure(figsize=(12, 8))
plt.plot(np.arange(len(p_values)), p_values, marker='o', linestyle='-', color='g', linewidth=2, label='p-values')

# Highlight significant p-values
plt.scatter(significant_indices, p_values[significant_indices], color='red', label='Significant (p < 0.05)', zorder=5)

plt.axhline(y=p_value_threshold, color='r', linestyle='--', label=f'p-value threshold = {p_value_threshold}')
plt.xlabel('Sound Index')
plt.ylabel('p-value')
plt.title('Paired T-test p-values with Significance Threshold')
plt.grid(True)
plt.legend()
plt.show()


