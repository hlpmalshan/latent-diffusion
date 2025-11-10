import os

# Folders containing the images
# train_dir = "data/cifar-100-python/train_images"
# test_dir = "data/cifar-100-python/test_images"

train_dir = "data/celeba/train"
test_dir = "data/celeba/test"
val_dir = "data/celeba/val"

# Output txt files
# train_txt = "data/cifar-100-python/train_paths.txt"
# test_txt = "data/cifar-100-python/test_paths.txt"

train_txt = "data/celeba/train.txt"
test_txt = "data/celeba/test.txt"
val_txt = "data/celeba/val.txt"

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
save_paths(val_dir, val_txt)

print("Completed")
