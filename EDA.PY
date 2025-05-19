import os
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = "chest_xray"
sets = ["train", "val", "test"]
categories = ["NORMAL", "PNEUMONIA"]

# Store counts for visualization
data_counts = {"Dataset": [], "Category": [], "Count": []}

for dataset in sets:
    for category in categories:
        folder_path = os.path.join(data_dir, dataset, category)
        count = len(os.listdir(folder_path))
        print(f"{dataset}/{category}: {count} images")

        # Store for visualization
        data_counts["Dataset"].append(dataset)
        data_counts["Category"].append(category)
        data_counts["Count"].append(count)

# Visualize dataset distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=data_counts["Dataset"], y=data_counts["Count"], hue=data_counts["Category"])
plt.title("Dataset Distribution")
plt.ylabel("Number of Images")
plt.show()
