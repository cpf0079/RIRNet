# -*- coding: utf-8 -*-

import os
import csv


def get_label(dir, address):
    with open(dir, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        print(len(result))

    with open(address, 'a') as q:
        for i in range(len(result)-1):
            q.write(result[i+1][0] + ' '+ result[i+1][1] + '\n')


if __name__ == "__main__":
    dir = "KoNViD_1k_mos.csv"
    address = "label.txt"
    get_label(dir, address)


