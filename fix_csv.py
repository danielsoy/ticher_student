import pandas as pd

# Load the CSV file
df = pd.read_csv('data/carpet/carpet.csv')

# Set label=0 for 'good' samples
df.loc[df['image_name'].str.startswith('good_'), 'label'] = 0

# Remove ground truth entries for 'good' samples
df.loc[df['image_name'].str.startswith('good_'), 'gt_name'] = ''

# Set type='train' for first 70% of 'good' samples
good_samples = df[df['image_name'].str.startswith('good_')]
train_count = int(len(good_samples) * 0.7)
good_indices = good_samples.index.tolist()
train_indices = good_indices[:train_count]
test_indices = good_indices[train_count:]

# Set types
df.loc[train_indices, 'type'] = 'train'
df.loc[test_indices, 'type'] = 'test'
df.loc[~df['image_name'].str.startswith('good_'), 'type'] = 'test'

# Save the modified CSV file
df.to_csv('data/carpet/carpet.csv', index=False)

print("CSV file updated successfully!")