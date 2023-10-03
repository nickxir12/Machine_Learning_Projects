# %%
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
import matplotlib.pyplot as plt
from scipy.linalg import svd

# %%
# Read data
filename = "../MyData/HTRU_2.csv"
df = pd.read_csv(filename)

raw_data = df.values

cols = range(0, 9)
X = raw_data[:, cols]
# Column of platelets has to be divided with 1000,mistake in data

attributeNames = np.asarray(df.columns[cols])

classLabels = raw_data[:, -1]  # -1 takes the last column

classNames = np.unique(classLabels)

classDict = dict(zip(classNames, range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape

C = len(classNames)

classDict = dict(zip(range(1,8), classNames))

# %%
import seaborn as sns

# Finding outliers
# Keep only the first 8 columns
# changing the lines below depending on what we want
# either combined/pulsar/non pulsar samples
filtered_df = df[df.iloc[:, -1] == 0]
cols_to_keep = df.columns[:8]
x = filtered_df[cols_to_keep].values

# Compute the mean of each feature. 
mean = np.mean(x, axis=0)

# Compute the standard deviation of each feature 
standard_deviation = np.std(x, axis=0, ddof=1)

# Compute the minimum of each feature 
minimum = np.min(x, axis=0)

# Compute the maximum of each feature 
maximum = np.max(x, axis=0)

# Compute the three quartiles (25°, 50° e 75°) of each feature
quartile_25 = np.percentile(x, 25, axis=0)
quartile_50 = np.percentile(x, 50, axis=0)
quartile_75 = np.percentile(x, 75, axis=0)

# Compute the variance of each feature
variances = np.var(x, axis=0, ddof=1)

# Print the results of each feature
for i in range(7):
    print(f"Minimum: {minimum[i+1]}")
    print(f"Maximum: {maximum[i+1]}")
    print(f"25° percentile: {quartile_25[i+1]}")
    print(f"50° percentile (Mediana): {quartile_50[i+1]}")
    print(f"75° percentile: {quartile_75[i+1]}")
    print(f"Variances: {variances[i+1]}")
    print()
    
    
plt.figure(figsize=(12, 8))

# Iterate through each feature and create a boxplot
for i in range(1,8):  # Skip the first and last columns
    plt.subplot(3, 3, i)  # Create subplots in a 3x3 grid
    sns.boxplot(x=x[:, i])
    plt.title(f"Box Plot for {attributeNames[i]}")
    plt.xlabel(attributeNames[i])

plt.tight_layout()  # Adjust subplot spacing for better layout
plt.show()  

# %%
#Plot 2 attributes
i = 0       #first attribute
j = 4       #fifth attribute

f = figure()
title('Pulsar Data')

for c in classNames:
    # select indices belonging to class c:
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)


legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()

# %%
from scipy import stats

Y = (X - np.ones((N, 1)) * X.mean(axis=0))/np.std(X, axis=0)

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()

# %%
pcs = [0,1,2,3]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b', 'y']
bw = .1
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Pulsar data: PCA Component Coefficients')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# %%
# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Pulsar data: 1st and 2nd PCA Components')
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

i = 0
j = 1

for index in range(6):
  ax = axes[index]
  for c in classNames:
      # select indices belonging to class c:
      class_mask = y==c
      ax.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
      ax.set_ylabel('PC{0}'.format(j+1), fontsize=12)
      ax.set_xlabel('PC{0}'.format(i+1), fontsize=12)
  #Do not repeat plots
  j+=1
  if j == 4:
    i = i+1
    if (i<3): j = i+1
    else: j = 3

plt.tight_layout()
show()

# %%
# Indices of the principal components to be plotted
i = 0
j = 1
k = 2

# Plot PCA of the data
f = figure()
title('Pulsar data: first 3 PCA Components')
ax = plt.axes(projection='3d')

for c in classNames:
    # select indices belonging to class c:
    class_mask = y==c
    ax.scatter(Z[class_mask,i],Z[class_mask,k],Z[class_mask,j])
    # plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
ax.set_xlabel('PC{0}'.format(i+1))
ax.set_ylabel('PC{0}'.format(j+1))
ax.set_zlabel('PC{0}'.format(k+1))

# Output result to screen
show()

# %%
# Create a 3x3 grid of subplots for histograms
fig, axes = plt.subplots(3, 3, figsize=(10, 8))

# Flatten the axes array for easier iteration
axes = axes.flatten()
beans = 100

pulsars = X[X[:, -1] == 1]
non_pulsars = X[X[:, -1] == 0]

# Create histograms for each subplot
for i in range(9):
    ax = axes[i]
    ax.hist(X[:,i], bins=beans, color='orange')
    # uncomment next 2 lines to see comparison of histograms between classes
    # ax.hist(pulsars[:,i], bins=beans, color='blue')
    # ax.hist(non_pulsars[:,i], bins=beans, color='red')
    ax.set_title(attributeNames[i])

# Add a title for the entire grid of histograms
fig.suptitle(f'Attribute Histograms(bins = {beans})', fontsize=16)

# Adjust the spacing between subplots
plt.tight_layout()
# Display the plot
plt.show()

# %%
# Create a 3x3 grid of subplots for histograms
fig, axes = plt.subplots(1, 4, figsize=(15, 3))

# Flatten the axes array for easier iteration
axes = axes.flatten()
beans = 100

pulsars = X[X[:, -1] == 1]
non_pulsars = X[X[:, -1] == 0]
n = 0
# Create histograms for each subplot
for i in range(8):
    ax = axes[n]
    if (attributeNames[i] == 'Profile_mean' or attributeNames[i] == 'Profile_stdev' or attributeNames[i] == 'DM_stdev' or attributeNames[i] == 'DM_skewness'):
        n += 1
        ax.hist(pulsars[:,i], bins=beans, color='blue', density=True)
        ax.hist(non_pulsars[:,i], bins=beans, color='red', density=True)
        ax.set_title(attributeNames[i])
    if (n == 4): break

# Add a title for the entire grid of histograms
fig.suptitle(f'Attribute Histograms(bins = {beans})', fontsize=16)

# Adjust the spacing between subplots
plt.tight_layout()
# Display the plot
plt.show()

# %%
#       For statistical results-Part 2 of Report

for column_index in range(X.shape[1]):
    column_mean = np.mean(X[:, column_index])
    column_std = np.std(X[:, column_index], ddof=1)
    column_median = np.median(X[:, column_index])
    column_range = np.max(X[:, column_index]) - np.min(X[:, column_index])
    print(f"Attribute {attributeNames[column_index]}")
    print(f"Mean : {column_mean}")
    print(f"Standar Deviation : {column_std}")
    print(f"Median : {column_median}")
    print(f"Range: {column_range}")
    print()

# %%
# Calculate the correlation matrix
correlation_matrix = np.corrcoef(Y[:, :8], rowvar=False)

# Create a heatmap of the correlation matrix
mask = np.ones_like(correlation_matrix, dtype=bool)

for i in range(8):
    for j in range(8):
        if i < j:
          mask[i, j] = False
        else:
          pass

plt.figure(figsize=(10, 8))

heatmap = plt.imshow(np.ma.masked_array(correlation_matrix, mask=~mask), cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)

plt.colorbar(heatmap, label='Correlation Coefficient')
plt.title('Correlation Matrix Heatmap')
plt.xticks(range(len(attributeNames)-1), attributeNames[:-1], fontsize=14)
plt.yticks(range(len(attributeNames)-1), attributeNames[:-1], fontsize=14)
for i in range(8):
    for j in range(8):
        if i <= j:
          plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha='center', va='center', color='white')
          ax.text(3,3, correlation_matrix[3,3],fontsize=24)
        elif i < j:
          pass
        else:
          plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha='center', va='center', color='black')
          ax.text(3,3, correlation_matrix[3,3],fontsize=24)

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# %%
print(correlation_matrix)


