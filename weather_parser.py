from collections import defaultdict
import csv

FILES = {
    '2012-01-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201201.txt',
    '2012-02-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201202.txt',
    '2012-03-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201203.txt',
    '2012-04-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201204.txt',
    '2012-05-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201205.txt',
    '2012-06-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201206.txt',
    '2012-07-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201207.txt',
    '2012-08-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201208.txt',
    '2012-09-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201209.txt',
    '2012-10-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201210.txt',
    '2012-11-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201211.txt',
    '2012-12-': 'weather-data/LJUBLJANA_-_BEZIGRAD_201212.txt'
}


def parse():
    data = defaultdict(float)
    for file in FILES:
        f = open(FILES[file], "r", encoding="utf-8")
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for line in reader:
            key = (file + line[0]) if len(line[0]) > 1 else (file + '0' + line[0])
            data[key] = float(line[2]) if line[2] != '' else 0.
        f.close()

    return data
