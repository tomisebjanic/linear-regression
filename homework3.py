import gzip
import csv
from collections import defaultdict
import lpputils
import scipy
import scipy.sparse as sp
import linear
import numpy as np
import time
import datetime

# Registration          0
# Driver ID	            1
# Route	                2
# Route Direction	    3
# Route description	    4
# First station	        5
# Departure time	    6
# Last station	        7
# Arrival time          8

# Morning rush hour: 7:00 - 9:00
# Afternoon rush hour: 15:00 - 17:00
# Busy days: mon, tue, wed, thu, fri

# Best model contains params:
#   - Departure time
#   - Driver ID
#   - Busy day

FORMAT = "%Y-%m-%d %H:%M:%S.%f"
BUSY = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


class Homework3:

    def __init__(self):
        parsed_data = self.parse_training_data()
        self.y = parsed_data[0]
        self.train_data = parsed_data[1]

    def linekey(self, d):
        return tuple(d[2:4])

    def get_time(self, t):
        time_obj = datetime.datetime.strptime(t, FORMAT)
        return float(time_obj.hour*3600 + time_obj.minute*60 + time_obj.second)

    def parse_training_data(self):
        f = gzip.open("ucni-podatki/train.csv.gz", "rt", encoding="utf-8")
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        dataY = defaultdict(list)
        dataX = defaultdict(list)
        raw = [d for d in reader]
        print("===Parsing train_data===")
        i = 465562
        for d in raw:
            print(i)
            id_trip = tuple(d[2:4])
            departure_time_obj = datetime.datetime.strptime(d[6], FORMAT)
            params = [self.get_time(d[6]), float(d[1])]
            # if 7 <= departure_time_obj.hour <= 9:
            #     params.append(1.)
            # else:
            #     params.append(0.)
            # if 15 <= departure_time_obj.hour <= 17:
            #     params.append(1.)
            # else:
            #     params.append(0.)

            if departure_time_obj.strftime("%A") in BUSY:
                params.append(1.)
            else:
                params.append(0.)

            dataY[id_trip].append(float(lpputils.tsdiff(d[8], d[6])))   # Trip Time
            dataX[id_trip].append(params)
            i -= 1

        return [dataY, dataX]

    def do_it(self):
        models = defaultdict(list)
        for m in self.train_data:
            X = np.array(self.train_data[m])
            Xsp = scipy.sparse.csr_matrix(X)
            y = np.array(self.y[m])
            lr = linear.LinearLearner(lambda_=0.05)
            models[m] = lr(Xsp, y)

        f = gzip.open("ucni-podatki/test.csv.gz", "rt", encoding="utf-8")
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        raw = [d for d in reader]
        f.close()
        fo_name = "result_" + datetime.datetime.fromtimestamp(timestamp=int(time.time())).strftime("%Y-%m-%d-%H_%M") + ".txt"
        fo = open(fo_name, "wt")

        for line in raw:
            id_trip = tuple(line[2:4])
            dt = line[6]
            departure_time_obj = datetime.datetime.strptime(dt, FORMAT)
            params = [self.get_time(line[6]), float(line[1])]

            # if 7 <= departure_time_obj.hour <= 9:
            #     params.append(1.)
            # else:
            #     params.append(0.)
            # if 15 <= departure_time_obj.hour <= 17:
            #     params.append(1.)
            # else:
            #     params.append(0.)

            if departure_time_obj.strftime("%A") in BUSY:
                params.append(1.)
            else:
                params.append(0.)

            trip_time = models[id_trip](np.array(params))
            arrival_time = lpputils.tsadd(dt, trip_time)

            print('Departure:', dt, 'Trip Len:', trip_time, 'Arrival:', arrival_time)

            fo.write(arrival_time + "\n")
        fo.close()





hw = Homework3().do_it()
