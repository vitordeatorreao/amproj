"""Base IO code for reading from a file or the standard input"""

from .dataset import Dataset


def read_dataset(reader,
                 separator=',',
                 comment=";",
                 headers=[],
                 include_headers=True):
    """Reads a dataset from a file input

    Parameters
    ----------
    reader : str or file
        The path to the file containing the dataset or the file input stream.

    separator : str, optional, default: ','
        Character or string that separates the features in the lines.

    comment : str, optional, default: ';'
        Character or string that represents the beginning of a comment line.

    headers : list<str>, optional, deafult: []
        List of headers for the resulting dataset

    include_headers : boolean, optional, default: True
        Whether or not headers are provided for the features

    Returns
    -------
    dataset : Dataset object
        The resulting Dataset instance which should be an exact memory
        representation of the data in the input.
    """
    created_reader = False
    if type(reader) == str:
        reader = open(reader, "r")
        created_reader = True
    dataset = Dataset(headers)  # by default, starts with no headers
    for line in reader:
        line = line.strip()
        if line.startswith(comment) or len(line) == 0:
            continue
        if len(headers) == 0 and include_headers:
            for header in line.split(separator):
                header = header.strip()
                if len(header) == 0:
                    continue
                headers.append(header)
            dataset = Dataset(headers)
            continue
        datapoint = [d.strip() for d in line.split(separator)]
        dataset.add_datapoint(datapoint)
    if created_reader:
        reader.close()
    return dataset


def read_from_data_file_with_headers(filepath):
    """Reads the dataset from a .data file provided by UCI repository
    with header information.

    Parameters
    ----------
    filepath : str
        The path to the file to be used as input

    Returns
    -------
    dataset : Dataset object
        The resulting Dataset, which should be an exact memory representation
        of the data in the file.
    """
    reader = open(filepath, "r")
    headers = ["CLASS"]
    while len(headers) < 2:
        line = reader.readline().strip()
        if line.startswith(";") or len(line) == 0:
            continue
        for header in line.split(","):
            header = header.strip()
            headers.append(header)
    dataset = read_dataset(reader, separator=",", comment=";",
                           headers=headers, include_headers=False)
    reader.close()
    return dataset
