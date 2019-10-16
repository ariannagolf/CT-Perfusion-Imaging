import pydicom
import numpy as np
import os
import random
from datetime import datetime
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

#x range = 26, 97
#y range = 27, 106

compiled_arr = [] #Final array to hold all input/output data from all patients
arr_input = []
arr_ttp = []
arr_rbf = []
arr_rbv = []
arr_mtt = []
arr_tmax = []
arr_length = []
j = 0

# FOR ALL DATA 50 points per file in 54 folders
# numFiles = 54 
# numPoints = 50

# # FOR ALL DATA 100 points per file in 54 folders
numFiles = 54 
numPoints = 100

# # FOR ALL DATA 200 points per file in 54 folders
# numFiles = 54 
# numPoints = 200

###############################
while (j < numFiles): #Go through all patients (this should change to 51)
    val_arr = []
    i = 0
    # point_dict holds points that we use so we don't reuse the same ones
    point_dict =  {}

    if (j < 9):
        dcm_folder = '/CS168_Project/Data/00{}'.format(j+1) + '/'  #this path should be changed based on where the data is located

    else:
        dcm_folder = '/CS168_Project/Data/0{}'.format(j + 1) + '/' #this path should be changed based on where the data is located

    while (i < numPoints): #Find 200 random points per patient
        if (i+1) % 10 == 0:
            print('reading file {} point {}'.format(j+1, i+1))
        # get slices
        TTP_entries = os.listdir(dcm_folder + 'IMAGES/SRS1/')
        scans_index = []
        for entry in TTP_entries:
            if (entry[-4:] != ".out"):
                ds = pydicom.dcmread(dcm_folder + 'IMAGES/SRS1/' + entry)
                scans_index.append(round(ds.SliceLocation, 1))

        #find a random x, y, z value
        x = random.randint(26, 97)
        y = random.randint(27, 106)
        z = random.randint(0, len(scans_index)-1)

        # if this combindation has already been used, make a new random combination
        while (x,y, z) in point_dict:
            x = random.randint(26, 97)
            y = random.randint(27, 106)
            z = random.randint(0, len(scans_index)-1)

        point_dict[(x,y, z)] = "True"
        x_loc = x
        y_loc = y
        sliceNum = z

        ########################################################################
       
        # get perfusion data
        time_index = []  # store converted time (microseconds)
        time_intervals = []  # store interval between Acquisition times (in microseconds)
        perfusion_entries = os.listdir(dcm_folder + 'study/')
        perfusion_index = dict()
        first = True
        firsttime = 0
        for entry in perfusion_entries:
            if (entry[-3:] == "dcm"):
                ds = pydicom.dcmread(dcm_folder + 'study/' + entry)
                # print(ds.SliceLocation)
                if ds.SliceLocation == scans_index[sliceNum]:
                    a = datetime.strptime(ds.AcquisitionTime, '%H%M%S.%f')
                    td = a.hour * 3600000000 + a.minute * 60000000 + a.second * 1000000 + a.microsecond
                    if first == True:
                        firsttime = td
                        first = False
                    cur = td - firsttime
                    if ds.SliceLocation not in perfusion_index:
                        perfusion_index[cur] = [] 
                    dcm_scan = ds.pixel_array
                    perfusion_index[cur].append(dcm_scan[y_loc][x_loc])

        temp_arr = []
        for it in sorted(perfusion_index.keys()):
            temp_arr.append((it, perfusion_index[it]))
        interpDict = dict()
        xp = []
        fp = []

        for time, val in temp_arr:
            xp.append(time)
            fp.append(val[0])
        x = range(0, xp[-1], xp[-1]/100)
        interp = np.interp(x, xp, fp)
        arr_length.append(len(interp))


        # items of interpDict are a tuple (slice location, interpolated array 0-33 for acq)

        ########################################################################

        k = 0
        while (k < 4):
            folder = 'IMAGES/SRS{}'.format(k+1) + '/'
            entries = os.listdir(dcm_folder + folder)
            first = entries[1]
            ds = pydicom.dcmread(dcm_folder + 'IMAGES/SRS{}/'.format(k + 1) + '/' + first)
            tag = ds[0x1313,0x1014].value
            if (tag == 'TTP'):
                TTP_entries = entries
                for entry in TTP_entries:
                    if (entry[-4:] != ".out"):
                        ds = pydicom.dcmread(dcm_folder + 'IMAGES/SRS{}/'.format(k+1) + '/' + entry)
                        if round(ds.SliceLocation, 1) == scans_index[sliceNum]:
                            dcm_scan = ds.pixel_array
                            TTP_val = dcm_scan[y_loc][x_loc]
                # gives the TTP value at the slice/x/y

                ########################################################################
            elif (tag == 'rBF'):
                rBF_entries = entries
                for entry in rBF_entries:
                    if (entry[-4:] != ".out"):
                        ds = pydicom.dcmread(dcm_folder + 'IMAGES/SRS{}/'.format(k+1) + '/' + entry)
                        if round(ds.SliceLocation, 1) == scans_index[sliceNum]:
                            dcm_scan = ds.pixel_array
                            rBF_val = dcm_scan[y_loc][x_loc]
                # gives the rBF value at the slice/x/y

                ########################################################################
            elif (tag == 'rBV'):
                rBV_entries = entries
                for entry in rBV_entries:
                    if (entry[-4:] != ".out"):
                        ds = pydicom.dcmread(dcm_folder + 'IMAGES/SRS{}/'.format(k+1) + '/' + entry)
                        if round(ds.SliceLocation, 1) == scans_index[sliceNum]:
                            dcm_scan = ds.pixel_array
                            rBV_val = dcm_scan[y_loc][x_loc]
                # gives the rBV value at the slice/x/y

                ########################################################################
            elif (tag == 'MTT'):
                MTT_entries = entries
                for entry in MTT_entries:
                    if (entry[-4:] != ".out"):
                        ds = pydicom.dcmread(dcm_folder + 'IMAGES/SRS{}/'.format(k+1) + '/' + entry)
                        #print(entry)
                        #print(ds.SliceLocation)
                        if round(ds.SliceLocation, 1) == scans_index[sliceNum]:
                            dcm_scan = ds.pixel_array
                            MTT_val = dcm_scan[y_loc][x_loc]
                # gives the MTT value at the slice/x/y
                ########################################################################
            elif (tag == 'TMAX'):
                TMAX_entries = entries
                for entry in TMAX_entries:
                    if (entry[-4:] != ".out"):
                        ds = pydicom.dcmread(dcm_folder + 'IMAGES/SRS{}/'.format(k+1) + '/' + entry)
                        if round(ds.SliceLocation, 1) == scans_index[sliceNum]:
                            dcm_scan = ds.pixel_array
                            TMAX_val = dcm_scan[y_loc][x_loc]
            k = k + 1
                # gives the TMAX value at the slice/x/y

                ########################################################################
        #create an entry to hold input values (x,y,z) and output values (TTP, rBF, rBV, MTT,TMAX)
        #we can use this data and enter it to scikit.learn
        arr_input.append(interp)
        arr_ttp.append(TTP_val)
        arr_rbf.append(rBF_val)
        arr_rbv.append(rBV_val)
        arr_mtt.append(MTT_val)
        i = i+1
    j = j + 1

min_length = min(arr_length)

index = arr_length.index(min_length)

tempi = [t[:min_length] for t in arr_input]

arr_input = tempi

# #ALL DATA ARRAYS 
# input_train = arr_input[0:1700]
# input_test = arr_input[1700:2700]
# output_ttp_train = arr_ttp[0:1700]
# output_ttp_test = arr_ttp[1700:2700]
# output_rbv_train = arr_rbv[0:1700]
# output_rbv_test = arr_rbv[1700:2700]
# output_rbf_train = arr_rbf[0:1700]
# output_rbf_test = arr_rbf[1700:2700]
# output_mtt_train = arr_mtt[0:1700]
# output_mtt_test = arr_mtt[1700:2700]

#ALL DATA ARRAYS for 100 points
input_train = arr_input[0:3400]
input_test = arr_input[3400:5400]
output_ttp_train = arr_ttp[0:3400]
output_ttp_test = arr_ttp[3400:5400]
output_rbv_train = arr_rbv[0:3400]
output_rbv_test = arr_rbv[3400:5400]
output_rbf_train = arr_rbf[0:3400]
output_rbf_test = arr_rbf[3400:5400]
output_mtt_train = arr_mtt[0:3400]
output_mtt_test = arr_mtt[3400:5400]

# #ALL DATA ARRAYS for 200 points
# input_train = arr_input[0:6800]
# input_test = arr_input[6800:10800]
# output_ttp_train = arr_ttp[0:6800]
# output_ttp_test = arr_ttp[6800:10800]
# output_rbv_train = arr_rbv[0:6800]
# output_rbv_test = arr_rbv[6800:10800]
# output_rbf_train = arr_rbf[0:6800]
# output_rbf_test = arr_rbf[6800:10800]
# output_mtt_train = arr_mtt[0:6800]
# output_mtt_test = arr_mtt[6800:10800]


## groups created for cross validation
groups = []
for groupNumber in range(numFiles):
    for p in range (numPoints):
        groups.append(groupNumber)
group_train = groups[0:len(input_train)]
gkf = GroupKFold(n_splits = len(input_train)/numPoints)

print("////////////////////K NEAREST NEIGHBORS////////////////////")

from sklearn.neighbors import KNeighborsClassifier
arr_knn_ttp_acc = []
arr_knn_rbf_acc = []
arr_knn_rbv_acc = []
arr_knn_mtt_acc = []

for k in range(1, len(input_train)/4, len(input_train)/40):
    knn_ttp = KNeighborsClassifier(n_neighbors=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_ttp_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_ttp_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_ttp_train[i])
        knn_ttp.fit(new_input_train, new_output_train)
        knn_ttp_pred = knn_ttp.predict(new_input_cv)
        knn_ttp_acc = metrics.accuracy_score(new_output_cv, knn_ttp_pred.round())
        temp_acc.append(knn_ttp_acc)
    arr_knn_ttp_acc.append(np.mean(temp_acc))

    knn_rbf = KNeighborsClassifier(n_neighbors=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbf_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbf_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbf_train[i])
        knn_rbf.fit(new_input_train, new_output_train)
        knn_rbf_pred = knn_rbf.predict(new_input_cv)
        knn_rbf_acc = metrics.accuracy_score(new_output_cv, knn_rbf_pred.round())
        temp_acc.append(knn_rbf_acc)
    arr_knn_rbf_acc.append(np.mean(temp_acc))

    knn_rbv = KNeighborsClassifier(n_neighbors=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbv_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbv_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbv_train[i])
        knn_rbv.fit(new_input_train, new_output_train)
        knn_rbv_pred = knn_rbv.predict(new_input_cv)
        knn_rbv_acc = metrics.accuracy_score(new_output_cv, knn_rbv_pred.round())
        temp_acc.append(knn_rbv_acc)
    arr_knn_rbv_acc.append(np.mean(temp_acc))

    knn_mtt = KNeighborsClassifier(n_neighbors=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_mtt_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_mtt_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_mtt_train[i])
        knn_mtt.fit(new_input_train, new_output_train)
        knn_mtt_pred = knn_mtt.predict(new_input_cv)
        knn_mtt_acc = metrics.accuracy_score(new_output_cv, knn_mtt_pred.round())
        temp_acc.append(knn_mtt_acc)
    arr_knn_mtt_acc.append(np.mean(temp_acc))



xs = range(0, len(input_train)/4, len(input_train)/40)
plt.figure(1)
plt.plot(xs, arr_knn_ttp_acc)
plt.savefig('1/knn_ttp_acc.png')
plt.figure(2)
plt.plot(xs, arr_knn_rbf_acc)
plt.savefig('1/knn_rbf_acc.png')
plt.figure(3)
plt.plot(xs, arr_knn_rbv_acc)
plt.savefig('1/knn_rbv_acc.png')
plt.figure(4)
plt.plot(xs, arr_knn_mtt_acc)
plt.savefig('1/knn_mtt_acc.png')

max_ttp_acc = max(arr_knn_ttp_acc)
index_max_ttp_acc = arr_knn_ttp_acc.index(max_ttp_acc)
max_rbf_acc = max(arr_knn_rbf_acc)
index_max_rbf_acc = arr_knn_rbf_acc.index(max_rbf_acc)
max_rbv_acc = max(arr_knn_rbv_acc)
index_max_rbv_acc = arr_knn_rbv_acc.index(max_rbv_acc)
max_mtt_acc = max(arr_knn_mtt_acc)
index_max_mtt_acc = arr_knn_mtt_acc.index(max_mtt_acc)

knn_ttp = KNeighborsClassifier(n_neighbors=(1 + (len(input_train)/40)*index_max_ttp_acc))
knn_ttp.fit(input_train, output_ttp_train)
knn_ttp_pred = knn_ttp.predict(input_test)
knn_ttp_acc = metrics.accuracy_score(output_ttp_test, knn_ttp_pred.round())
print("KNN TTP using {} neighbors".format(1 + (len(input_train)/40)*index_max_ttp_acc))
print(knn_ttp_acc)
ymax = max(output_ttp_test)
ymin = min(output_ttp_test)
mse = metrics.mean_squared_error(output_ttp_test, knn_ttp_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

knn_rbf = KNeighborsClassifier(n_neighbors=(1 + (len(input_train)/40)*index_max_rbf_acc))
knn_rbf.fit(input_train, output_rbf_train)
knn_rbf_pred = knn_rbf.predict(input_test)
knn_rbf_acc = metrics.accuracy_score(output_rbf_test, knn_rbf_pred.round())
print("KNN rBF using {} neighbors".format(1 + (len(input_train)/40)*index_max_rbf_acc))
print(knn_rbf_acc)
ymax = max(output_rbf_test)
ymin = min(output_rbf_test)
mse = metrics.mean_squared_error(output_rbf_test, knn_rbf_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

knn_rbv = KNeighborsClassifier(n_neighbors=(1 + (len(input_train)/40)*index_max_rbv_acc))
knn_rbv.fit(input_train, output_rbv_train)
knn_rbv_pred = knn_rbv.predict(input_test)
knn_rbv_acc = metrics.accuracy_score(output_rbv_test, knn_rbv_pred.round())
print("KNN rBV using {} neighbors".format(1 + (len(input_train)/40)*index_max_rbv_acc))
print(knn_rbv_acc)
ymax = max(output_rbv_test)
ymin = min(output_rbv_test)
mse = metrics.mean_squared_error(output_rbv_test, knn_rbv_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

knn_mtt = KNeighborsClassifier(n_neighbors=(1 + (len(input_train)/40)*index_max_mtt_acc))
knn_mtt.fit(input_train, output_mtt_train)
knn_mtt_pred = knn_mtt.predict(input_test)
knn_mtt_acc = metrics.accuracy_score(output_mtt_test, knn_mtt_pred.round())
print("KNN MTT using {} neighbors".format(1 + (len(input_train)/40)*index_max_mtt_acc))
print(knn_mtt_acc)
ymax = max(output_mtt_test)
ymin = min(output_mtt_test)
mse = metrics.mean_squared_error(output_mtt_test, knn_mtt_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

# #SVMs

print("////////////////////SVM linear////////////////////")

from sklearn import svm

svc_ttp = svm.SVC(kernel='linear')
svc_ttp.fit(input_train, output_ttp_train)
svc_ttp_pred = svc_ttp.predict(input_test)
print("SVM LINEAR TTP")
svc_ttp_acc = metrics.accuracy_score(output_ttp_test, svc_ttp_pred.round())
print(svc_ttp_acc)
ymax = max(output_ttp_test)
ymin = min(output_ttp_test)
mse = metrics.mean_squared_error(output_ttp_test, svc_ttp_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

svc_rbf = svm.SVC(kernel='linear')
svc_rbf.fit(input_train, output_rbf_train)
svc_rbf_pred = svc_rbf.predict(input_test)
print("SVM LINEAR rBF")
svc_rbf_acc = metrics.accuracy_score(output_rbf_test, svc_rbf_pred.round())
print(svc_rbf_acc)
ymax = max(output_rbf_test)
ymin = min(output_rbf_test)
mse = metrics.mean_squared_error(output_rbf_test, svc_rbf_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

svc_rbv = svm.SVC(kernel='linear')
svc_rbv.fit(input_train, output_rbv_train)
svc_rbv_pred = svc_rbv.predict(input_test)
print("SVM LINEAR rBV")
svc_rbv_acc = metrics.accuracy_score(output_rbv_test, svc_rbv_pred.round())
print(svc_rbv_acc)
ymax = max(output_rbv_test)
ymin = min(output_rbv_test)
mse = metrics.mean_squared_error(output_rbv_test, svc_rbv_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

svc_mtt = svm.SVC(kernel='linear')
svc_mtt.fit(input_train, output_mtt_train)
svc_mtt_pred = svc_mtt.predict(input_test)
print("SVM LINEAR mTT")
svc_mtt_acc = metrics.accuracy_score(output_mtt_test, svc_mtt_pred.round())
print(svc_mtt_acc)
ymax = max(output_mtt_test)
ymin = min(output_mtt_test)
mse = metrics.mean_squared_error(output_mtt_test, svc_mtt_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

print("////////////////////SVM rbf////////////////////")

arr_svc_ttp_acc_c = []
arr_svc_rbf_acc_c = []
arr_svc_rbv_acc_c = []
arr_svc_mtt_acc_c = []
arr_svc_ttp_acc_g = []
arr_svc_rbf_acc_g = []
arr_svc_rbv_acc_g = []
arr_svc_mtt_acc_g = []

C_range = 10.0 ** np.arange(-2, 2)
gamma_range = [0.15, 0.1, 0.05, 0.01]
print("fitting for ttp constants")
for c in C_range:
    svc_ttp = svm.SVC(C=c, kernel='rbf')
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_ttp_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_ttp_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_ttp_train[i])
        svc_ttp.fit(new_input_train, new_output_train)
        svc_ttp_pred = svc_ttp.predict(new_input_cv)
        svc_ttp_acc = metrics.accuracy_score(new_output_cv, svc_ttp_pred.round())
        temp_acc.append(svc_ttp_acc)
    arr_svc_ttp_acc_c.append(np.mean(temp_acc))
print("fitting for ttp gammas")
for g in gamma_range:
    svc_ttp = svm.SVC(kernel='rbf', gamma=g)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_ttp_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_ttp_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_ttp_train[i])
        svc_ttp.fit(new_input_train, new_output_train)
        svc_ttp_pred = svc_ttp.predict(new_input_cv)
        svc_ttp_acc = metrics.accuracy_score(new_output_cv, svc_ttp_pred.round())
        temp_acc.append(svc_ttp_acc)
    arr_svc_ttp_acc_g.append(np.mean(temp_acc))
print("fitting for rbf constants")
for c in C_range:
    svc_rbf = svm.SVC(C=c, kernel='rbf')
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbf_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbf_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbf_train[i])
        svc_rbf.fit(new_input_train, new_output_train)
        svc_rbf_pred = svc_rbf.predict(new_input_cv)
        svc_rbf_acc = metrics.accuracy_score(new_output_cv, svc_rbf_pred.round())
        temp_acc.append(svc_rbf_acc)
    arr_svc_rbf_acc_c.append(np.mean(temp_acc))
print("fitting for rbf gammas")
for g in gamma_range:
    svc_rbf = svm.SVC(kernel='rbf', gamma=g)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbf_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbf_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbf_train[i])
        svc_rbf.fit(new_input_train, new_output_train)
        svc_rbf_pred = svc_rbf.predict(new_input_cv)
        svc_rbf_acc = metrics.accuracy_score(new_output_cv, svc_rbf_pred.round())
        temp_acc.append(svc_rbf_acc)
    arr_svc_rbf_acc_g.append(np.mean(temp_acc))
print("fitting for rbv constants")
for c in C_range:
    svc_rbv = svm.SVC(C=c, kernel='rbf')
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbv_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbv_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbv_train[i])
        svc_rbv.fit(new_input_train, new_output_train)
        svc_rbv_pred = svc_rbv.predict(new_input_cv)
        svc_rbv_acc = metrics.accuracy_score(new_output_cv, svc_rbv_pred.round())
        temp_acc.append(svc_rbv_acc)
    arr_svc_rbv_acc_c.append(np.mean(temp_acc))
print("fitting for rbv gammas")
for g in gamma_range:
    svc_rbv = svm.SVC(kernel='rbf', gamma=g)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbv_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbv_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbv_train[i])
        svc_rbv.fit(new_input_train, new_output_train)
        svc_rbv_pred = svc_rbv.predict(new_input_cv)
        svc_rbv_acc = metrics.accuracy_score(new_output_cv, svc_rbv_pred.round())
        temp_acc.append(svc_rbv_acc)
    arr_svc_rbv_acc_g.append(np.mean(temp_acc))
print("fitting for mtt constants")
for c in C_range:
    svc_mtt = svm.SVC(C=c, kernel='rbf')
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_mtt_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_mtt_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_mtt_train[i])
        svc_mtt.fit(new_input_train, new_output_train)
        svc_mtt_pred = svc_mtt.predict(new_input_cv)
        svc_mtt_acc = metrics.accuracy_score(new_output_cv, svc_mtt_pred.round())
        temp_acc.append(svc_mtt_acc)
    arr_svc_mtt_acc_c.append(np.mean(temp_acc))
print("fitting for mtt gammas")
for g in gamma_range:
    svc_mtt = svm.SVC(kernel='rbf', gamma=g)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_mtt_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_mtt_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_mtt_train[i])
        svc_mtt.fit(new_input_train, new_output_train)
        svc_mtt_pred = svc_mtt.predict(new_input_cv)
        svc_mtt_acc = metrics.accuracy_score(new_output_cv, svc_mtt_pred.round())
        temp_acc.append(svc_mtt_acc)
    arr_svc_mtt_acc_g.append(np.mean(temp_acc))


xs_g = gamma_range
xs_c = C_range

plt.figure(5)
plt.plot(xs_g, arr_svc_ttp_acc_g)
plt.savefig('1/svc_ttp_acc_gamma.png')
plt.figure(6)
plt.plot(xs_c, arr_svc_ttp_acc_c)
plt.savefig('1/svc_ttp_acc_c.png')

plt.figure(7)
plt.plot(xs_g, arr_svc_rbf_acc_g)
plt.savefig('1/svc_rbf_acc_gamma.png')
plt.figure(8)
plt.plot(xs_c, arr_svc_rbf_acc_c)
plt.savefig('1/svc_rbf_acc_c.png')

plt.figure(9)
plt.plot(xs_g, arr_svc_rbv_acc_g)
plt.savefig('1/svc_rbv_acc_gamma.png')
plt.figure(10)
plt.plot(xs_c, arr_svc_rbv_acc_c)
plt.savefig('1/svc_rbv_acc_c.png')

plt.figure(11)
plt.plot(xs_g, arr_svc_mtt_acc_g)
plt.savefig('1/svc_mtt_acc_gamma.png')
plt.figure(12)
plt.plot(xs_c, arr_svc_mtt_acc_c)
plt.savefig('1/svc_mtt_acc_c.png')

max_ttp_acc_c = max(arr_svc_ttp_acc_c)
index_max_ttp_acc_c = arr_svc_ttp_acc_c.index(max_ttp_acc_c)
max_rbf_acc_c = max(arr_svc_rbf_acc_c)
index_max_rbf_acc_c = arr_svc_rbf_acc_c.index(max_rbf_acc_c)
max_rbv_acc_c = max(arr_svc_rbv_acc_c)
index_max_rbv_acc_c = arr_svc_rbv_acc_c.index(max_rbv_acc_c)
max_mtt_acc_c = max(arr_svc_mtt_acc_c)
index_max_mtt_acc_c = arr_svc_mtt_acc_c.index(max_mtt_acc_c)

max_ttp_acc_g = max(arr_svc_ttp_acc_g)
index_max_ttp_acc_g = arr_svc_ttp_acc_g.index(max_ttp_acc_g)
max_rbf_acc_g = max(arr_svc_rbf_acc_g)
index_max_rbf_acc_g = arr_svc_rbf_acc_g.index(max_rbf_acc_g)
max_rbv_acc_g = max(arr_svc_rbv_acc_g)
index_max_rbv_acc_g = arr_svc_rbv_acc_g.index(max_rbv_acc_g)
max_mtt_acc_g = max(arr_svc_mtt_acc_g)
index_max_mtt_acc_g = arr_svc_mtt_acc_g.index(max_mtt_acc_g)

svc_ttp = svm.SVC(C=C_range[index_max_ttp_acc_c], kernel='rbf', gamma=gamma_range[index_max_ttp_acc_g])
svc_ttp.fit(input_train, output_ttp_train)
svc_ttp_pred = svc_ttp.predict(input_test)
print("SVM TTP using c={} gamma={}".format(C_range[index_max_ttp_acc_c], gamma_range[index_max_ttp_acc_g]))
svc_ttp_acc = metrics.accuracy_score(output_ttp_test, svc_ttp_pred.round())
print(svc_ttp_acc)
ymax = max(output_ttp_test)
ymin = min(output_ttp_test)
mse = metrics.mean_squared_error(output_ttp_test, svc_ttp_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

svc_rbf = svm.SVC(C=C_range[index_max_rbf_acc_c], kernel='rbf', gamma=gamma_range[index_max_rbf_acc_g])
svc_rbf.fit(input_train, output_rbf_train)
svc_rbf_pred = svc_rbf.predict(input_test)
print("SVM rBF using c={}, gamma={}".format(C_range[index_max_rbf_acc_c], gamma_range[index_max_rbf_acc_g]))
svc_rbf_acc = metrics.accuracy_score(output_rbf_test, svc_rbf_pred.round())
print(svc_rbf_acc)
ymax = max(output_rbf_test)
ymin = min(output_rbf_test)
mse = metrics.mean_squared_error(output_rbf_test, svc_rbf_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

svc_rbv = svm.SVC(C=C_range[index_max_rbv_acc_c], kernel='rbf', gamma=gamma_range[index_max_rbv_acc_g])
svc_rbv.fit(input_train, output_rbv_train)
svc_rbv_pred = svc_rbv.predict(input_test)
print("SVM rBV using c={}, gamma={}".format(C_range[index_max_rbv_acc_c], gamma_range[index_max_rbv_acc_g]))
svc_rbv_acc = metrics.accuracy_score(output_rbv_test, svc_rbv_pred.round())
print(svc_rbv_acc)
ymax = max(output_rbv_test)
ymin = min(output_rbv_test)
mse = metrics.mean_squared_error(output_rbv_test, svc_rbv_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

svc_mtt = svm.SVC(C=C_range[index_max_mtt_acc_c], kernel='rbf', gamma=gamma_range[index_max_mtt_acc_g])
svc_mtt.fit(input_train, output_mtt_train)
svc_mtt_pred = svc_mtt.predict(input_test)
print("SVM mTT using c={}, gamma={}".format(C_range[index_max_mtt_acc_c], gamma_range[index_max_mtt_acc_g]))
svc_mtt_acc = metrics.accuracy_score(output_mtt_test, svc_mtt_pred.round())
print(svc_mtt_acc)
ymax = max(output_mtt_test)
ymin = min(output_mtt_test)
mse = metrics.mean_squared_error(output_mtt_test, svc_mtt_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)



### auto

svc_ttp = svm.SVC(kernel='rbf')
svc_ttp.fit(input_train, output_ttp_train)
svc_ttp_pred = svc_ttp.predict(input_test)
print("SVM TTP no paramerters specified")
svc_ttp_acc = metrics.accuracy_score(output_ttp_test, svc_ttp_pred.round())
print(svc_ttp_acc)
ymax = max(output_ttp_test)
ymin = min(output_ttp_test)
mse = metrics.mean_squared_error(output_ttp_test, svc_ttp_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

svc_rbf = svm.SVC(kernel='rbf')
svc_rbf.fit(input_train, output_rbf_train)
svc_rbf_pred = svc_rbf.predict(input_test)
print("SVM rBF no paramerters specified")
svc_rbf_acc = metrics.accuracy_score(output_rbf_test, svc_rbf_pred.round())
print(svc_rbf_acc)
ymax = max(output_rbf_test)
ymin = min(output_rbf_test)
mse = metrics.mean_squared_error(output_rbf_test, svc_rbf_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

svc_rbv = svm.SVC(kernel='rbf')
svc_rbv.fit(input_train, output_rbv_train)
svc_rbv_pred = svc_rbv.predict(input_test)
print("SVM rBV no paramerters specified")
svc_rbv_acc = metrics.accuracy_score(output_rbv_test, svc_rbv_pred.round())
print(svc_rbv_acc)
ymax = max(output_rbv_test)
ymin = min(output_rbv_test)
mse = metrics.mean_squared_error(output_rbv_test, svc_rbv_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

svc_mtt = svm.SVC(kernel='rbf')
svc_mtt.fit(input_train, output_mtt_train)
svc_mtt_pred = svc_mtt.predict(input_test)
print("SVM mTT no paramerters specified")
svc_mtt_acc = metrics.accuracy_score(output_mtt_test, svc_mtt_pred.round())
print(svc_mtt_acc)
ymax = max(output_mtt_test)
ymin = min(output_mtt_test)
mse = metrics.mean_squared_error(output_mtt_test, svc_mtt_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)


print("////////////////////ENSEMBLE ADABOOST////////////////////")

from sklearn.ensemble import AdaBoostClassifier
arr_ada_ttp_acc = []
arr_ada_rbf_acc = []
arr_ada_rbv_acc = []
arr_ada_mtt_acc = []
totalTrain = len(input_train) - numPoints
totalTrain = 30
for k in range(1, totalTrain, totalTrain/5):
    print("fitting for ttp {} estimators".format(k))
    ada_ttp = AdaBoostClassifier(n_estimators=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_ttp_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_ttp_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_ttp_train[i])
        ada_ttp.fit(new_input_train, new_output_train)
        ada_ttp_pred = ada_ttp.predict(new_input_cv)
        ada_ttp_acc = metrics.accuracy_score(new_output_cv, ada_ttp_pred.round())
        temp_acc.append(ada_ttp_acc)
    arr_ada_ttp_acc.append(np.mean(temp_acc))

    print("fitting for rbf {} estimators".format(k))
    ada_rbf = AdaBoostClassifier(n_estimators=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbf_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbf_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbf_train[i])
        ada_rbf.fit(new_input_train, new_output_train)
        ada_rbf_pred = ada_rbf.predict(new_input_cv)
        ada_rbf_acc = metrics.accuracy_score(new_output_cv, ada_rbf_pred.round())
        temp_acc.append(ada_rbf_acc)
    arr_ada_rbf_acc.append(np.mean(temp_acc))

    print("fitting for rbv {} estimators".format(k))
    ada_rbv = AdaBoostClassifier(n_estimators=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbv_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbv_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbv_train[i])
        ada_rbv.fit(new_input_train, new_output_train)
        ada_rbv_pred = ada_rbv.predict(new_input_cv)
        ada_rbv_acc = metrics.accuracy_score(new_output_cv, ada_rbv_pred.round())
        temp_acc.append(ada_rbv_acc)
    arr_ada_rbv_acc.append(np.mean(temp_acc))

    print("fitting for mtt {} estimators".format(k))
    ada_mtt = AdaBoostClassifier(n_estimators=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_mtt_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_mtt_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_mtt_train[i])
        ada_mtt.fit(new_input_train, new_output_train)
        ada_mtt_pred = ada_mtt.predict(new_input_cv)
        ada_mtt_acc = metrics.accuracy_score(new_output_cv, ada_mtt_pred.round())
        temp_acc.append(ada_mtt_acc)
    arr_ada_mtt_acc.append(np.mean(temp_acc))



xs = range(0, totalTrain, totalTrain/5)
plt.figure(13)
plt.plot(xs, arr_ada_ttp_acc)
plt.savefig('1/ada_ttp_acc.png')
plt.figure(14)
plt.plot(xs, arr_ada_rbf_acc)
plt.savefig('1/ada_rbf_acc.png')
plt.figure(15)
plt.plot(xs, arr_ada_rbv_acc)
plt.savefig('1/ada_rbv_acc.png')
plt.figure(16)
plt.plot(xs, arr_ada_mtt_acc)
plt.savefig('1/ada_mtt_acc.png')

max_ttp_acc = max(arr_ada_ttp_acc)
index_max_ttp_acc = arr_ada_ttp_acc.index(max_ttp_acc)
max_rbf_acc = max(arr_ada_rbf_acc)
index_max_rbf_acc = arr_ada_rbf_acc.index(max_rbf_acc)
max_rbv_acc = max(arr_ada_rbv_acc)
index_max_rbv_acc = arr_ada_rbv_acc.index(max_rbv_acc)
max_mtt_acc = max(arr_ada_mtt_acc)
index_max_mtt_acc = arr_ada_mtt_acc.index(max_mtt_acc)

ada_ttp = AdaBoostClassifier(n_estimators=(1 + (totalTrain/5)*index_max_ttp_acc))
ada_ttp.fit(input_train, output_ttp_train)
ada_ttp_pred = ada_ttp.predict(input_test)
ada_ttp_acc = metrics.accuracy_score(output_ttp_test, ada_ttp_pred.round())
print("ADABOOST TTP using {} estimators".format(1 + (totalTrain/5)*index_max_ttp_acc))
print(ada_ttp_acc)
ymax = max(output_ttp_test)
ymin = min(output_ttp_test)
mse = metrics.mean_squared_error(output_ttp_test, ada_ttp_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

ada_rbf = AdaBoostClassifier(n_estimators=(1 + (totalTrain/5)*index_max_rbf_acc))
ada_rbf.fit(input_train, output_rbf_train)
ada_rbf_pred = ada_rbf.predict(input_test)
ada_rbf_acc = metrics.accuracy_score(output_rbf_test, ada_rbf_pred.round())
print("ADABOOST rBF using {} estimators".format(1 + (totalTrain/5)*index_max_rbf_acc))
print(ada_rbf_acc)
ymax = max(output_rbf_test)
ymin = min(output_rbf_test)
mse = metrics.mean_squared_error(output_rbf_test, ada_rbf_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

ada_rbv = AdaBoostClassifier(n_estimators=(1 + (totalTrain/5)*index_max_rbv_acc))
ada_rbv.fit(input_train, output_rbv_train)
ada_rbv_pred = ada_rbv.predict(input_test)
ada_rbv_acc = metrics.accuracy_score(output_rbv_test, ada_rbv_pred.round())
print("ADABOOST rBV using {} estimators".format(1 + (totalTrain/5)*index_max_rbv_acc))
print(ada_rbv_acc)
ymax = max(output_rbv_test)
ymin = min(output_rbv_test)
mse = metrics.mean_squared_error(output_rbv_test, ada_rbv_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

ada_mtt = AdaBoostClassifier(n_estimators=(1 + (totalTrain/5)*index_max_mtt_acc))
ada_mtt.fit(input_train, output_mtt_train)
ada_mtt_pred = ada_mtt.predict(input_test)
ada_mtt_acc = metrics.accuracy_score(output_mtt_test, ada_mtt_pred.round())
print("ADABOOST MTT using {} estimators".format(1 + (totalTrain/5)*index_max_mtt_acc))
print(ada_mtt_acc)
ymax = max(output_mtt_test)
ymin = min(output_mtt_test)
mse = metrics.mean_squared_error(output_mtt_test, ada_mtt_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

print("////////////////////ENSEMBLE RANDOM FORESTS////////////////////")

from sklearn.ensemble import RandomForestClassifier
arr_rf_ttp_acc = []
arr_rf_rbf_acc = []
arr_rf_rbv_acc = []
arr_rf_mtt_acc = []

totalTrain = 20

for k in range(1, totalTrain, totalTrain/5):
    print("fitting for {} estimators".format(k))
    rf_ttp = RandomForestClassifier(n_estimators=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_ttp_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_ttp_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_ttp_train[i])
        rf_ttp.fit(new_input_train, new_output_train)
        rf_ttp_pred = rf_ttp.predict(new_input_cv)
        rf_ttp_acc = metrics.accuracy_score(new_output_cv, rf_ttp_pred.round())
        temp_acc.append(rf_ttp_acc)
    arr_rf_ttp_acc.append(np.mean(temp_acc))

    rf_rbf = RandomForestClassifier(n_estimators=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbf_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbf_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbf_train[i])
        rf_rbf.fit(new_input_train, new_output_train)
        rf_rbf_pred = rf_rbf.predict(new_input_cv)
        rf_rbf_acc = metrics.accuracy_score(new_output_cv, rf_rbf_pred.round())
        temp_acc.append(rf_rbf_acc)
    arr_rf_rbf_acc.append(np.mean(temp_acc))

    rf_rbv = RandomForestClassifier(n_estimators=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbv_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbv_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbv_train[i])
        rf_rbv.fit(new_input_train, new_output_train)
        rf_rbv_pred = rf_rbv.predict(new_input_cv)
        rf_rbv_acc = metrics.accuracy_score(new_output_cv, rf_rbv_pred.round())
        temp_acc.append(rf_rbv_acc)
    arr_rf_rbv_acc.append(np.mean(temp_acc))

    rf_mtt = RandomForestClassifier(n_estimators=k)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_mtt_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_mtt_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_mtt_train[i])
        rf_mtt.fit(new_input_train, new_output_train)
        rf_mtt_pred = rf_mtt.predict(new_input_cv)
        rf_mtt_acc = metrics.accuracy_score(new_output_cv, rf_mtt_pred.round())
        temp_acc.append(rf_mtt_acc)
    arr_rf_mtt_acc.append(np.mean(temp_acc))



xs = range(0, totalTrain, totalTrain/5)
plt.figure(17)
plt.plot(xs, arr_rf_ttp_acc)
plt.savefig('1/rf_ttp_acc.png')
plt.figure(18)
plt.plot(xs, arr_rf_rbf_acc)
plt.savefig('1/rf_rbf_acc.png')
plt.figure(19)
plt.plot(xs, arr_rf_rbv_acc)
plt.savefig('1/rf_rbv_acc.png')
plt.figure(20)
plt.plot(xs, arr_rf_mtt_acc)
plt.savefig('1/rf_mtt_acc.png')

max_ttp_acc = max(arr_rf_ttp_acc)
index_max_ttp_acc = arr_rf_ttp_acc.index(max_ttp_acc)
max_rbf_acc = max(arr_rf_rbf_acc)
index_max_rbf_acc = arr_rf_rbf_acc.index(max_rbf_acc)
max_rbv_acc = max(arr_rf_rbv_acc)
index_max_rbv_acc = arr_rf_rbv_acc.index(max_rbv_acc)
max_mtt_acc = max(arr_rf_mtt_acc)
index_max_mtt_acc = arr_rf_mtt_acc.index(max_mtt_acc)

rf_ttp = RandomForestClassifier(n_estimators= index_max_ttp_acc * (totalTrain/5))
rf_ttp.fit(input_train, output_ttp_train)
rf_ttp_pred = rf_ttp.predict(input_test)
rf_ttp_acc = metrics.accuracy_score(output_ttp_test, rf_ttp_pred.round())
print("RANDOM FORESTS TTP using {} estimators".format(index_max_ttp_acc * (totalTrain/5)))
print(rf_ttp_acc)
ymax = max(output_ttp_test)
ymin = min(output_ttp_test)
mse = metrics.mean_squared_error(output_ttp_test, rf_ttp_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)


rf_rbf = RandomForestClassifier(n_estimators=index_max_rbf_acc * (totalTrain/5))
rf_rbf.fit(input_train, output_rbf_train)
rf_rbf_pred = rf_rbf.predict(input_test)
rf_rbf_acc = metrics.accuracy_score(output_rbf_test, rf_rbf_pred.round())
print("RANDOM FORESTS rBF using {} estimators".format(index_max_rbf_acc * (totalTrain/5)))
print(rf_rbf_acc)
ymax = max(output_rbf_test)
ymin = min(output_rbf_test)
mse = metrics.mean_squared_error(output_rbf_test, rf_rbf_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

rf_rbv = RandomForestClassifier(n_estimators=index_max_rbv_acc * (totalTrain/5))
rf_rbv.fit(input_train, output_rbv_train)
rf_rbv_pred = rf_rbv.predict(input_test)
rf_rbv_acc = metrics.accuracy_score(output_rbv_test, rf_rbv_pred.round())
print("RANDOM FORESTS rBV using {} estimators".format(index_max_rbv_acc * (totalTrain/5)))
print(rf_rbv_acc)
ymax = max(output_rbv_test)
ymin = min(output_rbv_test)
mse = metrics.mean_squared_error(output_rbv_test, rf_rbv_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

rf_mtt = RandomForestClassifier(n_estimators=index_max_mtt_acc * (totalTrain/5))
rf_mtt.fit(input_train, output_mtt_train)
rf_mtt_pred = rf_mtt.predict(input_test)
rf_mtt_acc = metrics.accuracy_score(output_mtt_test, rf_mtt_pred.round())
print("RANDOM FORESTS MTT using {} estimators".format(index_max_mtt_acc * (totalTrain/5)))
print(rf_mtt_acc)
ymax = max(output_mtt_test)
ymin = min(output_mtt_test)
mse = metrics.mean_squared_error(output_mtt_test, rf_mtt_pred.round())
rmse = np.sqrt(mse)
nrmse = rmse/(ymax - ymin)
print(mse, rmse, nrmse)

