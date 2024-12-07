import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, norm

# Load the normalization file once
norm_mats = scipy.io.loadmat('E:/courses/IISC/all_MATfiles/all_MATfiles/all-20240626T062008Z-001/all/EPE/EPE.mat')
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
sounds = ["EPE-sub01", "EPE-sub02", "EPE-sub03", "EPE-sub04", "EPE-sub05", "EPE-sub06", "EPE-sub07", "EPE-sub08", "EPE-sub09", "EPE-sub010",
          "EPE-sub011", "EPE-sub012", "EPE-sub013", "EPE-sub014", "EPE-sub015", "EPE-sub016", "EPE-sub017", "EPE-sub018", "EPE-sub019", "EPE-sub020",
          "EPE-sub021", "EPE-sub022", "EPE-sub023", "EPE-sub024", "EPE-sub025", "EPE-sub026", "EPE-sub027", "EPE-sub028", "EPE-sub029", "EPE-sub030",
          "EPE-sub031", "EPE-sub032", "EPE-sub033", "EPE-sub034", "EPE-sub035", "EPE-sub036", "EPE-sub037", "EPE-sub038", "EPE-sub039", "EPE-sub040",
          "EPE-sub042", "EPE-sub043", "EPE-sub044", "EPE-sub045", "EPE-sub046", "EPE-sub047", "EPE-sub048", "EPE-sub049", "EPE-sub050",
          "EPE-sub051", "EPE-sub052", "EPE-sub053", "EPE-sub054", "EPE-sub055", "EPE-sub056", "EPE-sub057", "EPE-sub058", "EPE-sub059", "EPE-sub060",
          "EPE-sub061", "EPE-sub062", "EPE-sub063", "EPE-sub064", "EPE-sub065", "EPE-sub066", "EPE-sub067", "EPE-sub068", "EPE-sub069", "EPE-sub070",
          "EPE-sub071", "EPE-sub072", "EPE-sub073", "EPE-sub074", "EPE-sub075"]

for i, sound in enumerate(sounds):
    mats = scipy.io.loadmat('E:/courses/IISC_working/EPE/EPE/' + sound + '.mat')
    
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
plt.title('Normalized Euclidean Distances Between Corresponding UL and LL Points Across All Sounds with Wilson Intervals(EPE)')
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


