import os
import tempfile
import pytest
import sys

# Add parent directory to Python path
from create_labels import create_labels

@pytest.fixture
def input_file_path():
    return 'test/test_create_labels.json'

@pytest.fixture
def labels_file_path():
    return 'test/test_create_labels.csv'

def test_create_labels(input_file_path, labels_file_path):
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, 'test.csv')
        # Call the create_labels function with test input and output files
        create_labels(input_file_path, output_file)

        # Check that the output file was created
        assert os.path.isfile(output_file)

        # Check that the output file is not empty
        assert os.path.getsize(output_file) > 0

        # Check that the output file has the correct format
        with open(output_file, 'r') as f:
            lines = f.readlines()
            # Check header
            assert lines[0] == 'id1,id2,label\n'
            # Check that each line has three values separated by commas
            for line in lines[1:]:
                assert len(line.split(',')) == 3

        # Check that the output file matches the labels file in the test directory
        with open(labels_file_path, 'r') as f1, open(output_file, 'r') as f2:
            # for each line in the labels file
            for line1 in f1:
                # read the corresponding line in the output file
                line2 = f2.readline()
                # check that the lines match
                assert line1 == line2
