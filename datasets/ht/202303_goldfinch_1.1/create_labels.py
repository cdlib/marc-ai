"""
This script reads an input file containing MARC records in JSON format and compares the 'cid' field of every two
records. It then writes a label (either 1 or 0) to an output file, indicating whether the 'cid' field is the same
or different between the two records.

Usage:
    python create_labels_file.py input_file [--output_file output_file]

Arguments:
    input_file      The path to the input file containing MARC records in JSON format, one record per line.
    
Optional Arguments:
    --output_file   The path to the output file. If not specified, a new file will be created with the same name
                    as the input file but with '_labels.csv' appended to the end.

Example:
    python create_labels_file.py input.json --output_file output.csv
"""

import argparse
import json
import pymarc

def create_labels(input_file, output_file=None):
    """
    Create labels for a dataset by comparing the 'cid' field of every two MARC records in the input file.
    Writes the resulting labels (either 1 or 0) to the output file.
    
    Args:
        input_file (str): The path to the input file.
        output_file (str, optional): The path to the output file. If not specified, a new file will be created
                                     with the same name as the input file but with '_labels.csv' appended to the end.
        
    Returns:
        None
    """
    # Initialize variables
    record1_id, record2_id, record1_cid, record2_cid, label = [None] * 5
    
    # Open the input file
    with open(input_file, 'rb') as f:
        # Open the output file
        with open(output_file or (input_file.replace('.json', '_labels.csv')), 'w') as g:
            # Write header to output file
            g.write("id1,id2,label\n")
            # For each line in the input file
            for line_num, line in enumerate(f):
                # read line as a MARC record
                reader = pymarc.JSONReader(line.decode('utf-8'))
                for record in reader:
                    record_id = record.get_fields('001')[0].value()
                    record_cid = record.get_fields('CID')[0].value()
                    # Read the record pair   
                    if line_num % 2 == 0 and not record1_id:
                        record1_id, record1_cid = record_id, record_cid
                    elif line_num % 2 == 1 and not record2_id:
                        record2_id, record2_cid = record_id, record_cid
                # If record1 and record2 exist, compare the 'cid' field and write to output file
                if record1_cid and record2_cid:
                    label = int(record1_cid == record2_cid)
                    g.write(f"{record1_id},{record2_id},{label}\n")
                    # reset variables
                    record1_id, record2_id, record1_cid, record2_cid, label = [None] * 5

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create labels for a dataset')
    parser.add_argument('input_file', help='The path to the input file')
    parser.add_argument('--output_file', help='The path to the output file')
    args = parser.parse_args()
    # Check for output file and input/output file name conflicts
    if args.output_file == args.input_file:
        print("Input file cannot be the same as output file")
        exit()
    # Call create_labels_file function with command-line arguments
    create_labels(args.input_file, args.output_file)
