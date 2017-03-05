import io, os, csv, zipfile
from datetime import datetime
import numpy as np

def getDictFromCsv(csvInput, en_coding=None):
    """from a csv file, return a dict. key is the header, value is a list of the all the cells under this header."""
    dict = {}
    with open(csvInput, 'rU') as csvInput:
        reader = csv.DictReader(csvInput)
        fields = reader.fieldnames
        for row in reader:
            for field in fields:
                a = dict.get(field)
                if not a:
                    a = [row.get(field)]
                else:
                    a.append(row.get(field))
                dict.update({field: a})
    return dict


def getDictFromCsv_2(csvInput):
    """from a csv file, return a dict containing all the columns except for the header. only works for 2 column sheets, while the 1st column is key and 2nd column as value."""
    dict = {}
    with open(csvInput, 'rU') as csvInput:
        reader = csv.DictReader(csvInput)
        fields = reader.fieldnames
        for row in reader:
            dict.update({row.get(fields[0]): row.get(fields[1])})
    return dict


def zipDir(path, zipfile_handle):
    for root, dirs, files in os.walk(path):
        for file in files:
            zipfile_handle.write(os.path.join(root,file))


def make_zipfile(output_filename, source_dir):
    # relroot = os.path.abspath(os.path.join(source_dir, os.pardir))
    relroot = os.path.abspath(source_dir)
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zip:
        for root, dirs, files in os.walk(source_dir):
            # add directory (needed for empty dirs)
            if os.path.relpath(root, relroot) != '.':
                zip.write(root, os.path.relpath(root, relroot))
            for file in files:
                filename = os.path.join(root, file)
                if os.path.isfile(filename): # regular files only
                    arcname = os.path.join(os.path.relpath(root, relroot), file)
                    zip.write(filename, arcname)


def createXML(filename, displayname):
    template = """<?xml version="1.0" encoding="utf-8" ?>
<MailboxMetadata>
	<DisplayName>""" + displayname + """</DisplayName>
	<LastImportDate>""" + str(datetime.now()) + """</LastImportDate>
</MailboxMetadata>
"""
    with open(filename, 'w') as output:
        output.write(template)


def split_ndarray(array, repetition, round):
    total = len(array)
    bin = total / repetition
    start = round * bin
    end = (round + 1) * bin
    split_list = np.split(array, [start, end])
    test = split_list[1]
    split_list.pop(1)
    left_over = [a for a in split_list if a.__len__() != 0]
    if len(left_over) == 1:
        training = left_over[0]
    else:
        training = np.concatenate([left_over[0], left_over[1]])
    return training, test


def saveDict2Csv(input_dict, output_csv):
    with open(output_csv, 'wb') as csvOut:
        fields = input_dict.keys()
        writer = csv.DictWriter(csvOut, fieldnames= fields)
        writer.writeheader()
        writer.writerow(input_dict)
    return output_csv