% Define the new attribute names
newAttributeNames = {'Profile_mean', 'Profile_stdev', 'Profile_skewness', 'Profile_kurtosis', ...
                     'DM_mean', 'DM_stdev', 'DM_skewness', 'DM_kurtosis', 'class'};

% Define the path to your CSV file

scriptName = mfilename('fullpath');
scriptDirectory = fileparts(scriptName);
csvFile = fullfile(scriptDirectory, '..', 'MyData', 'HTRU_2.csv');

% Read data from the CSV file
data = readmatrix(csvFile, 'HeaderLines', 1);

% Assuming that the last column in the CSV file contains the class labels
% and the rest are features
X = data(:, 1:end-1); % Features
y = data(:, end);     % Class labels

% Replace the attribute names in the cell array
attributeNames = newAttributeNames;
M = size(X, 2);  % Number of features
N = size(X, 1);  % Number of samples

scriptName = mfilename('fullpath');
scriptDirectory = fileparts(scriptName);
matFile = fullfile(scriptDirectory, 'HTRU_2.mat');

save(matFile, 'attributeNames', 'M', 'N', 'X', 'y');



