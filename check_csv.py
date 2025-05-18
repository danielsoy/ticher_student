import pandas as pd

# Load the CSV file
df = pd.read_csv('data/carpet/carpet.csv')

# Display the first 5 rows
print("First 5 rows:")
print(df.head(5))

# Display statistics
print("\nStatistics:")
print(f"Total entries: {len(df)}")
print(f"Normal samples (label=0): {len(df[df['label'] == 0])}")
print(f"Anomalous samples (label=1): {len(df[df['label'] == 1])}")
print(f"Training samples (type='train'): {len(df[df['type'] == 'train'])}")
print(f"Testing samples (type='test'): {len(df[df['type'] == 'test'])}")

# Check for potential issues
issues = []
if len(df[df['label'] == 0]) == 0:
    issues.append("No normal samples (label=0)")
if len(df[df['type'] == 'train']) == 0:
    issues.append("No training samples (type='train')")
if any(df.loc[df['label'] == 0, 'gt_name'].str.strip() != ''):
    issues.append("Some normal samples have ground truth masks")

if issues:
    print("\nIssues found:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("\nNo issues found. The CSV file looks good!")