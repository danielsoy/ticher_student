import pandas as pd

# Load the CSV file
df = pd.read_csv('data/carpet/carpet.csv')

# Print a sample of normal entries with ground truth masks to understand the issue
normal_with_gt = df[(df['label'] == 0) & (df['gt_name'].str.strip() != '')]
if len(normal_with_gt) > 0:
    print("Examples of normal samples with ground truth masks:")
    print(normal_with_gt.head())

# Force empty string for gt_name where label is 0
df.loc[df['label'] == 0, 'gt_name'] = ''

# Save the modified CSV file
df.to_csv('data/carpet/carpet.csv', index=False)

print("CSV file updated successfully - removed ground truth masks from normal samples!")