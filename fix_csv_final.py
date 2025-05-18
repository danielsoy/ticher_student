import pandas as pd

# Load the CSV file
df = pd.read_csv('data/carpet/carpet.csv')

# Remove ground truth entries for normal samples (label=0)
df.loc[df['label'] == 0, 'gt_name'] = ''

# Save the modified CSV file
df.to_csv('data/carpet/carpet.csv', index=False)

print("CSV file updated successfully - removed ground truth masks from normal samples!")