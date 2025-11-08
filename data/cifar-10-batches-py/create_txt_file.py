import os

def write_txt_file(data_dir, out_file):
    with open(out_file, "w") as f:
        for name in sorted(os.listdir(data_dir)):
            if name.endswith(".png"):
                label = name.split("_")[2]  # since we saved as prefix_idx_label_filename
                # abs_path = os.path.abspath(os.path.join(data_dir, name))    # for absolute path
                abs_path = os.path.join(data_dir, name)  # for relative path
                f.write(abs_path + "\n")

write_txt_file("train", "train.txt")
write_txt_file("test", "test.txt")