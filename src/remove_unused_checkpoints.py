import glob
import os


def remove_unused_checkpoints(input_folder='../outputs/'):
    subfolders = [directory for directory in os.listdir(input_folder) if os.path.isdir(input_folder + directory)]
    for folder in subfolders:
        remove_unused_checkpoints(input_folder + folder + '/')

    files = glob.glob(input_folder + '/*.h5')
    files.sort(key=os.path.getmtime)
    for file in files[:-1]:
        os.remove(file)


if __name__ == "__main__":
    remove_unused_checkpoints()
