import os
import numpy as np

def convert_linemod_to_new_format(path_to_linemod):
# recursively step through all labels and append camera intrinsics to each label

    linemod_dict = {'fx': 572.4114, 'fy': 573.5704, 'u0': 325.2611, 'v0': 242.0489, 'width': 640, 'height': 480} 

    for root, dirs, files in os.walk(path_to_linemod):
        for file in files:
            if file.endswith(".txt") and 'train' not in file and 'test' not in file and 'val' not in file:
                print(os.path.join(root, file))

                with open(os.path.join(root, file), 'r+') as f:
                    # iterate over all the lines and append intrinsics to the end of the file
                    lines = f.readlines()
                    for i, line in enumerate(lines):

                        lines[i] = line.rstrip() + ' ' + str(linemod_dict['fx']) + ' ' + str(linemod_dict['fy']) + ' ' + str(linemod_dict['width']) + ' ' + str(linemod_dict['height']) + ' ' +  str(linemod_dict['u0']) + ' ' + str(linemod_dict['v0']) + ' ' + str(linemod_dict['width']) + ' ' + str(linemod_dict['height']) + '\n'
                        
                    # overwrite the file with the new content
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()

def fix_image_paths(path_to_linemod):
    # recursively step through all test, val and train text files and fix image paths

    for root, dirs, files in os.walk(path_to_linemod):
        for file in files:
            if file.endswith(".txt") and 'train' in file or 'test' in file or 'val' in file:
                print(os.path.join(root, file))
                # get object name
                object_name = os.path.basename(root)
                with open(os.path.join(root, file), 'r+') as f:
                    # iterate over all the lines and append intrinsics to the end of the file
                    lines = f.readlines()
                    for i, line in enumerate(lines):

                        lines[i] = line.replace('LINEMOD/'+ object_name+ '/JPEGImages/', '')
                        
                    # overwrite the file with the new content
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()


# parser for command line arguments
# main function
                    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert linemod to new format')
    parser.add_argument('--path_to_linemod', type=str, help='path to linemod dataset')
    
    args = parser.parse_args()

    convert_linemod_to_new_format(args.path_to_linemod)
    fix_image_paths(args.path_to_linemod)