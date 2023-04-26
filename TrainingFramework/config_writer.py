"""
This file is meant to be used to generate new 
config files automatically so that I don't have to do it
myself. 

Created by: Andreas Robertson
Contact: arobertson38@gatech.edu
"""
import os
import argparse

def transform_value(value):
    """
    This method transforms the values that you are passing
    in into their native state.

    '0.001' -> '0.001'
    'true' -> 'true'
    'false' -> 'false'
    'default' / any string -> "itself"
    """
    if all(val.isdigit() or val == '.' for val in value):
        # its a number
        return value
    elif value.lower() == 'false' or value.lower() == 'true':
        # its a boolean
        return value.lower()
    else:
        # its any type of string. 
        return '"' + value + '"'

def listtodict(in_list):
    """
    sorts in list into a dictionary
    """
    in_dict = {}
    for sequence in in_list:
        assert len(sequence) == 3, "Only 1 category is supported."
        if in_dict.keys().__contains__(sequence[0]):
            assert not in_dict[sequence[0]].keys().__contains__(sequence[1]), \
                    "You already defined {sequence[0]} -> {sequence[1]}."
            in_dict[sequence[0]][sequence[1]] = sequence[2]
        else:
            in_dict[sequence[0]] = {sequence[1]:sequence[2]}
    return in_dict

def new_filename(tag, in_dict):
    """
    Creates a new filename for the new config file. 

    :tag: the tag for the dataset. 
    :in_dict: the dictionary of things we are changing. 
    """
    filename = f"ref{tag}"
    for cat_key in in_dict.keys():
        for prop_key in in_dict[cat_key].keys():
            filename = filename + '_' + prop_key + in_dict[cat_key][prop_key].replace('.', 'd')
    filename = (filename + '.yml').replace('/', 'SLASH')
    return filename
            

def write_config(baseline_filename, filename, in_dict):
    """
    writes the config file. 

    :baseline_filename: the name of the baseline file we are drawing from. 
    :filename: the name of the filename we are writing to. 
    :in_dict: the dictionary of inputs we are writing to. 
    """
    # read the baseline file
    with open(baseline_filename, 'r') as f:
        lines = f.readlines()

    # writing a new file
    with open(os.path.join('./configs', filename), 'w') as f:
        cat_flag = True
        cat = 'default'
        while len(lines) > 0:
            line = lines[0]
            if len(line) == 1:
                # this is an empty line (just a new line character)
                f.write(line)
            elif not line[:2] == '  ':
                # this is a category line
                f.write(line)
                cat = line.split(':')[0]
                cat_flag = True if in_dict.keys().__contains__(cat) else False
            elif cat_flag:
                prop = line.split(':')[0].replace(' ', '')
                if in_dict[cat].keys().__contains__(prop):
                    # then we need to replace what exists with the inputted thing. 
                    f.write(
                        f"  {prop}: {transform_value(in_dict[cat][prop])}\n"
                            )
                else:
                    f.write(line)
            else:
                f.write(line)

            # remove the line
            del lines[0]

if __name__ == "__main__":
    # parsing
    parser = argparse.ArgumentParser(description='Write config files automatically.')
    parser.add_argument('--mode', type=str, default='exp', help='The default file to draw from.')
    parser.add_argument('--input', default=[], action='append', nargs='+', help='The arguments that should be changed. Should contain the full path. e.g., --in model dataset value')

    args = parser.parse_args()

    print(args.mode)
    print(args.input)

    in_dict = listtodict(args.input)
    print(in_dict)
    
    # checking the mode argument
    if args.mode.lower() == 'NBSA':
        filename = new_filename('NBSA', in_dict)
        write_config('./configs/BASELINES/cs1_NBSA_config.yml', filename, in_dict)

    elif args.mode.lower() == 'TI':
        filename = new_filename('TI', in_dict)
        write_config('./configs/BASELINES/cs2_TI_config.yml', filename, in_dict)

    else:
        raise NotImplementedError(f'Mode: {args.mode} is not supported.')
