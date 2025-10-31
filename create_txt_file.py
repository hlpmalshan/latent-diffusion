import os

# Folders containing the images
train_dir = "data/cifar-100-python/train_images"
test_dir = "data/cifar-100-python/test_images"

# Output txt files
train_txt = "data/cifar-100-python/train_paths.txt"
test_txt = "data/cifar-100-python/test_paths.txt"

# Function to save relative paths to a txt file
def save_paths(folder, output_file):
    # Get sorted list of files
    files = sorted(os.listdir(folder))
    with open(output_file, "w") as f:
        for file in files:
            path = os.path.join(folder, file)  # relative path
            f.write(f"{path}\n")

# Save paths
save_paths(train_dir, train_txt)
save_paths(test_dir, test_txt)

print(f"✅ Train image paths saved to {train_txt}")
print(f"✅ Test image paths saved to {test_txt}")
