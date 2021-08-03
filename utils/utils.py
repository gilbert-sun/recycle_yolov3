# -*- coding: utf-8 -*-
#!/usr/bin/env python3.6

import os
import glob

from exceptions import FileSizeError, FilesNumberError, FolderExistsError

IMG_FOLDER = ("images")
TXT_FOLDER = ("labels")
JSON_FOLDER = ("json_5cate", "json_7cate")
INVALIDATE = ()


def match_pattern(input_dirpath, input_patterns):
    output_lists = []
    for input_pattern in input_patterns:
        file_list_lower = glob.glob("{}{}{}".format(input_dirpath, os.sep, input_pattern))
        file_list_upper = glob.glob("{}{}{}".format(input_dirpath, os.sep, input_pattern.upper()))

        if len(file_list_lower) > 0 or len(file_list_upper) > 0:
             output_lists = file_list_lower + file_list_upper
        else:
             raise IndexError("input_pattern={} not matched any file".format("{}{}{}".format(input_dirpath, os.sep, input_pattern)))

    return output_lists



def mkdir_with_check(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        raise FolderExistsError("dirname already existed:　{}".format(dirname))


def list_dir(path):
    try:
        origin_list = os.listdir(path)
        for filename in origin_list:
            if filename.endswith(INVALIDATE):
                origin_list.remove(filename)
                print("removing {} from os.listdir".format(filename))
        return origin_list
    except Exception as e:
        print("list_dir error: ", e)
        raise



def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir():
            yield from scantree(entry.path)  # see below for Python 2.x
        else:
            yield entry


def search_filename(DIRPATH, GLOB_PATTERN):
    # print("DIRPATH: {}".format(DIRPATH))
    # print("GLOB_PATTERN: {}".format(GLOB_PATTERN))
    file_list = glob.glob(os.path.join(DIRPATH, GLOB_PATTERN))
    # print("file_list: {}".format(file_list))
    return file_list


def check_too_small(filepath, minsize):
    filesize = os.path.getsize(filepath)
    if filesize < minsize:
        # print("filesize too small, filepath: ", filepath)
        # print("filesize too small: ", filesize, "\n------------------------")
        return (True, filesize)
    return (False, filesize)


def list_dir(path):
    try:
        origin_list = os.listdir(path)
        for filename in origin_list:
            if filename.endswith(INVALIDATE):
                origin_list.remove(filename)
                print("removing {} from os.listdir".format(filename))
        return origin_list
    except Exception as e:
        raise BaseException("list_dir error: {}".format(e))


def remove_list(input_list):
    if len(input_list) > 3:
        print("You are going to remove > 3 files ?")
        # return
    for item in input_list:
        try:
            print("Remove this item: {}".format(item))
            os.remove(item)
        except:
            raise