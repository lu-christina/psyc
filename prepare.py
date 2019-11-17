# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

from glob2 import glob
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# 0 - male, 1 - female

# MAKE CSV LABELS FOR UTKFACE SET
# filename: [age]_[gender]_[race]_[date&time].jpg
# n = 23,709
def make_csv_utkface():
    df=pd.DataFrame(columns=["filename", "gender"])

    for image in glob("./datasets/utkface/*.jpg"):
        path = image.split('/')
        filename = path[-1]

        parse = filename.split('_')
        gender = parse[1]

        df = df.append({
            "filename": filename,
            "gender": gender
            }, ignore_index=True)

    df.to_csv('utkface_labels.csv', index=False)

# MAKE CSV LABELS FOR CFD SET
# filename: CFD_[race][gender]_**.jpg
# n = 1,208
def make_csv_cfd():
    df=pd.DataFrame(columns=["filename", "gender"])
    for image in glob("./datasets/cfd/*.jpg"):
        path = image.split('/')
        filename = path[-1]

        parse = filename.split('-')

        gender = 0
        if parse[1][1] == "F":
            gender = 1

        df = df.append({
            "filename": filename,
            "gender": gender
            }, ignore_index=True)

    df.to_csv('cfd_labels.csv', index=False)

# MOVE CFD FILES TO TEST (70%), TRAIN (15%), VALID (15%)
def partition_cfd():
    df=pd.read_csv('cfd_labels.csv')

    df.head()
    df.columns.values

    # get image labels for either gender
    male=df[df['gender']==0][['filename', 'gender']]
    female=df[df['gender']==1][['filename','gender']]

    m_train_X, m_test_X, train_y, test_y = train_test_split(male['filename'],male['gender'], random_state = 0, test_size=.3)
    f_train_X, f_test_X, train_y, test_y = train_test_split(female['filename'],female['gender'], random_state = 0, test_size=.3)


    origin_path='./datasets/cfd/'
    train_path='./datasets/cfd/train/'
    valid_path='./datasets/cfd/valid/'
    test_path='./datasets/cfd/test/'
    fm='female/'
    ml='male/'

    for file in m_train_X:
        os.rename(origin_path+file, train_path+ml+file)

    flip = 0
    for file in m_test_X:
        if flip % 2 == 0:
            os.rename(origin_path+file, valid_path+ml+file)
        else:
            os.rename(origin_path+file, test_path+ml+file)
        flip += 1

    for file in f_train_X:
        os.rename(origin_path+file, train_path+fm+file)

    flip = 0
    for file in f_test_X:
        if flip % 2 == 0:
            os.rename(origin_path+file, valid_path+fm+file)
        else:
            os.rename(origin_path+file, test_path+fm+file)
        flip += 1

# MOVE UTKFACE FILES TO TEST (70%), TRAIN (15%), VALID (15%)
def partition_utkface():
    df=pd.read_csv('utkface_labels.csv')

    df.head()
    df.columns.values

    # get image labels for either gender
    male=df[df['gender']==0][['filename', 'gender']]
    female=df[df['gender']==1][['filename','gender']]

    m_train_X, m_test_X, train_y, test_y = train_test_split(male['filename'],male['gender'], random_state = 0, test_size=.3)
    f_train_X, f_test_X, train_y, test_y = train_test_split(female['filename'],female['gender'], random_state = 0, test_size=.3)


    origin_path='./datasets/utkface/'
    train_path='./datasets/utkface/train/'
    valid_path='./datasets/utkface/valid/'
    test_path='./datasets/utkface/test/'
    fm='female/'
    ml='male/'

    for file in m_train_X:
        os.rename(origin_path+file, train_path+ml+file)

    flip = 0
    for file in m_test_X:
        if flip % 2 == 0:
            os.rename(origin_path+file, valid_path+ml+file)
        else:
            os.rename(origin_path+file, test_path+ml+file)
        flip += 1

    for file in f_train_X:
        os.rename(origin_path+file, train_path+fm+file)

    flip = 0
    for file in f_test_X:
        if flip % 2 == 0:
            os.rename(origin_path+file, valid_path+fm+file)
        else:
            os.rename(origin_path+file, test_path+fm+file)
        flip += 1

#
#
# me=read_and_prep_images(img_paths)
# preds=model.predict(me)
