import csv
import numpy as np
import datetime
from random import shuffle

def getData(file, test=False, perc=0.75, rand=False):
    """Reads data from csv file. Returns X, Y, X_test, Y_test.

    Arguments:
    file -- csv file to read in data from
    test -- Read training data (False). Read test data (True)
    perc -- Percent to split between train/test data
    rand -- Specify to shuffle the data
    """

    date = []
    year = []
    month = []
    category = []
    day = []
    district = [] 
    address = []
    lng = []
    lat = []
    with open(file, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date.append(row['Dates'])
            if not test:
                category.append(row['Category'])
            day.append(row['DayOfWeek'])
            district.append(row['PdDistrict'])
            address.append(row['Address'])
            lng.append(row['X'])
            lat.append(row['Y'])


    # format date
    for i in range(len(date)):
        d = date[i].split(' ')[0]
        t = date[i].split(' ')[1]
        hour = int(t.split(':')[0])
        minute = int(t.split(':')[1])
        date[i] = hour * 60 + minute
        year.append(int(d.split('-')[0]))
        month.append(int(d.split('-')[1]))

    # format category
    crimes = list(set(category))
    crimes = sorted(crimes)
    c = {}
    for i in range(len(crimes)):
        c[crimes[i]] = i
    for i in range(len(category)):
        category[i] = c[category[i]]

    # format day
    d = {'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,
         'Friday':5,'Saturday':6}
    for i in range(len(day)):
        day[i] = d[day[i]]

    # format district
    a = list(set(district))
    d = {}
    for i in range(len(a)):
        d[a[i]] = i
    for i in range(len(district)):
        district[i] = d[district[i]]

    # format address
    a = list(set(address))
    d = {}
    for i in range(len(a)):
        d[a[i]] = i
    for i in range(len(address)):
        address[i] = d[address[i]]

    # format lng/lat
    lng = [float(x) for x in lng]
    lat = [float(y) for y in lat]

    date_un = np.array(date)
    year_un = np.array(year)
    month_un = np.array(month)
    category_un = np.array(category)
    day_un = np.array(day)
    district_un = np.array(district)
    address_un = np.array(address)
    lng_un = np.array(lng)
    lat_un = np.array(lat)

    date = np.array(date)
    year = np.array(year)
    month = np.array(month)
    category = np.array(category)
    day = np.array(day)
    district = np.array(district)
    address = np.array(address)
    lng = np.array(lng)
    lat = np.array(lat)

    # select train/test set
    if test: 
        perc = 1.0
    m = len(date)
    indexes = range(m)
    if rand:
        shuffle(indexes)
    train_part = indexes[0 : int(perc * m)]
    test_part = indexes[int(perc * m) :]
    date = date[train_part]
    year = year[train_part]
    month = month[train_part]
    if not test:
        category = category[train_part]
    day = day[train_part]
    district = district[train_part]
    address = address[train_part]
    lng = lng[train_part]
    lat = lat[train_part]

    # form X and Y matrices
    X = np.concatenate(([date], [year], [month], [day], [district], [address], 
                        [lng], [lat]), axis=0).T
    
    Y = []
    for i in range(len(category_un)):
        temp = [0] * len(crimes)
        temp[category_un[i]] = 1
        Y.append(temp)
    Y = np.array(Y)

    X_test = np.concatenate(([date_un], [year_un], [month_un], [day_un], [district_un], [address_un], 
                             [lng_un], [lat_un]), axis=0).T

    if not test:
        X_test = X_test[test_part]
    else:
        X_test = []
    if not test:
        Y_test = Y[test_part]
    else:
        Y_test = []
    if not test:
        Y = Y[train_part]
    else:
        Y = []


    return {'X' : X, 'Y' : Y, 'X_test' : X_test, 'Y_test' : Y_test, 'crimes' : crimes}
