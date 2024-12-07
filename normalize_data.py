import pandas as pd
import json

# Load CVEFixes dataset
cvefixes_path = "Dataset/CVEFixes.csv"
cvefixes = pd.read_csv(cvefixes_path)

# Standardize CVEFixes dataset
cvefixes['label'] = cvefixes['safety'].apply(lambda x: 1 if x.lower() == 'vulnerable' else 0)
cvefixes_normalized = cvefixes[['code', 'label', 'language']]

# Load DiverseVul dataset
diversevul_path = "Dataset/diversevul/diversevul_20230702.json"
with open(diversevul_path, 'r') as file:
    diversevul_data = [json.loads(line) for line in file]

# Convert DiverseVul to DataFrame
diversevul_df = pd.DataFrame(diversevul_data)

# Standardize DiverseVul dataset
diversevul_df['label'] = diversevul_df['target']
diversevul_df['language'] = None
diversevul_normalized = diversevul_df[['func', 'label', 'language']].rename(columns={'func': 'code'})

# Combine both datasets
combined_dataset = pd.concat([cvefixes_normalized, diversevul_normalized], ignore_index=True)

# Inspect combined dataset
print(combined_dataset.head())