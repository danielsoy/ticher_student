import csv

# Read the CSV file
rows = []
with open('data/carpet/carpet.csv', 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # If label is 0, set gt_name to empty string
        if row['label'] == '0':
            row['gt_name'] = ''
        rows.append(row)

# Write the modified data back to the CSV file
with open('data/carpet/carpet.csv', 'w', newline='') as csvfile:
    fieldnames = ['image_name', 'gt_name', 'label', 'type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("CSV file updated successfully using direct CSV manipulation!")