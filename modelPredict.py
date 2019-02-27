import dill as pickle
import numpy as np
from sklearn import linear_model
from sklearn import utils, ensemble, metrics, svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import csv

a = open("./Data/data_final_standardized.pkl", "rb")
data,desc = pickle.load(a)
X,y = data

b = open("./Data/test_final_standardized.pkl", "rb")
data,desc = pickle.load(b)
Xt = data

Xlist = []
ylist = []
X_te = []
for i in X:
    Xlist.append(i.tolist())
for i in y:
    ylist.append(i.tolist())
for i in Xt:
    X_te.append(i.tolist())
X_tr = np.asarray(Xlist)
y_tr = np.asarray(ylist)
X_te = np.asarray(X_te)

# clf = linear_model.LinearRegression()
#clf = linear_model.Ridge(alpha=100)
#clf = KNeighborsRegressor(n_neighbors=1)
#clf = svm.SVR()
#clf = ensemble.AdaBoostRegressor()
#clf = ensemble.RandomForestRegressor()
clf = MLPRegressor()

# parameters = [
#{'C': [1, 10, 100, 1000], 'degree': [2,3,4,5], 'kernel': ['poly']},
#{'C': [1, 10, 100, 1000], 'gamma': [1.0,0.1,0.01,0.001], 'kernel': ['rbf']},
#{'C': [1, 10, 100, 1000], 'gamma': [.1,1,10,100], 'kernel': ['sigmoid']},
# {'C': [1], 'kernel': ['linear']},
# ]
# svr = sklearn.svm.SVR()
# clf = GridSearchCV(estimator=svr, param_grid=parameters, scoring='f1', cv=5)


clf.fit(X_tr, y_tr)
preds = clf.predict(X_te) # Predict Both

IDs = [18, 65, 90, 92, 107, 109, 174, 302, 371, 388, 406, 408, 442, 493, 533, 622, 626, 691, 698, 788, 950, 979, 1027, 1110, 1150, 1225, 1227, 1263, 1323, 1402, 1407, 1444, 1504, 1740, 1868, 1903, 1931, 1967, 2006, 2040, 2056, 2096, 2227, 2352, 2427, 2569, 2698, 2707, 2715, 2761, 2799, 3018, 3036, 3045, 3048, 3244, 3326, 3388, 3390, 3409, 3438, 3461, 3498, 3508, 3577, 3673, 3810, 3885, 3971, 4050, 4065, 4273, 4280, 4309, 4312, 4571, 4593, 4615, 4693, 4713, 4722, 4837, 4990, 5079, 5150, 5242, 5325, 5334, 5339, 5460, 5467, 5490, 5532, 5554, 5567, 5604, 5613, 5617, 5662, 5667, 5715, 5778, 5976, 6068, 6098, 6168, 6260, 6295, 6348, 6384, 6406, 6417, 6472, 6483, 6641, 6692, 6738, 6811, 6860, 6865, 6898, 6903, 6971, 6995, 6999, 7302, 7304, 7317, 7390, 7413, 7417, 7460, 7461, 7466, 7526, 7554, 7788, 7883, 8071, 8079, 8124, 8188, 8194, 8252, 8295, 8296, 8415, 8533, 8641, 8650, 8652, 8733, 8787, 8790, 8916, 8941, 8966, 9024, 9050, 9071, 9197, 9279, 9283, 9445, 9801, 9865, 9868, 9908, 9912, 9945, 9981, 10042, 10044, 10114, 10165, 10226, 10264, 10312, 10383, 10436, 10693, 10770, 10832, 10842, 10876, 10905, 11058, 11084, 11100, 11110, 11114, 11259, 11276, 11278, 11316, 11342, 11532, 11570, 11584, 11628, 11657, 11698, 11706, 11784, 11806, 11808, 11816, 11830, 11865, 11931, 12045, 12126, 12237, 12312, 12413, 12467, 12492, 12676, 12694, 12966, 12993, 13036, 13154, 13161, 13166, 13223, 13243, 13252, 13271, 13293, 13369, 13397, 13432, 13469, 13474, 13494, 13556, 13643, 13711, 13805, 13853, 13907, 13933, 13974, 13977, 13984, 14017, 14050, 14097, 14099, 14174, 14220, 14266, 14278, 14346, 14391, 14394, 14421, 14432, 14443, 14564, 14618, 14624, 14691, 14703, 14756, 14781, 14803, 14821, 14840, 14953, 15086, 15100, 15109, 15193, 15340, 15476, 15649, 15657, 15824, 15880, 15892, 15921, 15930, 15950, 15967, 16016, 16030, 16065, 16204, 16324, 16340, 16368, 16391, 16476, 16568, 16703, 16712, 16734, 16886, 16939, 16942, 16962, 17053, 17062, 17095, 17115, 17149, 17248, 17253, 17378, 17396, 17445, 17591, 17596, 17701, 17931, 18007, 18036, 18048, 18220, 18262, 18312, 18454, 18465, 18501, 18666, 18777, 18780, 18794, 18812, 18868, 18925, 18948, 18956, 18959, 18960, 18973, 18995, 19094, 19119, 19130, 19138, 19311, 19333, 19373, 19502, 19511, 19696, 19738, 19740, 19802, 19835, 19915, 19954, 20039, 20056, 20139, 20318, 20385, 20410, 20448, 20494, 20511, 20599, 20606, 20619, 20734, 20765, 20949, 20958, 20962, 21057, 21169, 21267, 21339, 21388, 21394, 21445, 21582, 21608, 21621, 21766, 21872, 21911, 22071, 22093, 22101, 22163, 22208, 22224, 22231, 22393, 22435, 22556, 22602, 22682, 22800, 22960, 22963, 23078, 23106, 23321, 23368, 23435, 23638, 23639, 23683, 23696, 23748, 23822, 23835, 23840, 23857, 24033, 24047, 24059, 24175, 24178, 24226, 24345, 24452, 24487, 24519, 24629, 24750, 24781, 24837, 24897, 24931, 24939, 24965, 24981, 25005, 25040, 25136, 25160, 25194, 25231, 25262, 25356, 25358, 25529, 25573, 25663, 25748, 25814, 25859, 25926, 25948, 26017, 26152, 26161, 26184, 26271, 26307, 26317, 26708, 26741, 26748, 26811, 26820, 26834, 26900, 26948, 26968, 27095, 27139, 27215, 27255, 27274, 27283, 27359, 27368, 27438, 27449, 27460, 27580, 27706, 27790, 27938, 28079, 28110, 28121, 28124, 28287, 28315, 28363, 28405, 28500, 28615, 28672, 28685, 28705, 28885, 28887, 28902, 29141, 29219, 29261, 29278, 29364, 29427, 29517, 29525, 29542, 29623, 29659, 29692, 29745, 29779, 29862, 29870, 29993, 30063, 30124, 30137, 30157, 30173, 30198, 30264, 30268, 30336, 30368, 30456, 30459, 30482, 30537, 30541, 30552, 30618, 30853, 30878, 30947, 30977, 31023, 31071, 31079, 31110, 31146, 31150, 31283, 31291, 31399, 31497, 31582, 31653, 31677, 31692, 31739, 31823, 31957, 31989, 32064, 32090, 32102, 32217, 32241, 32326, 32599, 32609, 32646, 32711, 32714, 32806, 32825, 32892, 32923, 32930, 33056, 33065, 33119, 33294, 33312, 33383, 33471, 33481, 33574, 33597, 33635, 33679, 33693, 33708, 33832, 33890, 33968, 33980, 34058, 34090, 34163, 34186, 34206, 34216, 34335, 34470, 34620, 34625, 34661, 34795, 35049, 35130, 35170, 35230, 35308, 35319, 35374, 35384, 35397, 35416, 35423, 35533, 35556, 35585, 35651, 35694, 35959, 35987, 35991, 35998, 36160, 36197, 36276, 36331, 36351, 36355, 36359, 36390, 36448, 36468, 36470, 36510, 36517, 36525, 36551, 36707, 36722, 36757, 36830, 36859, 36868, 36893, 36917, 36956, 36990, 37001, 37052, 37087, 37108, 37116, 37141, 37175, 37247, 37269, 37305, 37312, 37401, 37495, 37511, 37641, 37645, 37663, 37698, 37878, 37991, 38017, 38070, 38163, 38168, 38425, 38578, 38586, 38721, 38771, 38779, 38806, 38818, 38828, 38967, 38987, 39009, 39121, 39160, 39214, 39234, 39324, 39363, 39649, 39690, 39694, 39766, 39879, 39935, 39936, 40007, 40012, 40059, 40155, 40310, 40327, 40348, 40353, 40417, 40527, 40534, 40621, 40636, 40679, 40685, 40711, 40799, 40807, 40868, 40885, 40889, 40994, 41002, 41084, 41191, 41418, 41484, 41516, 41681, 41705, 41880, 41990, 41994, 42128, 42165, 42174, 42203, 42281, 42288, 42303, 42389, 42521, 42617, 42730, 42805, 42858, 42904, 42941, 42962, 42977, 43055, 43059, 43066, 43108, 43419, 43492, 43493, 43544, 43576, 43677, 44011, 44067, 44171, 44222, 44419, 44508, 44512, 44526, 44603, 44630, 44643, 44689, 44706, 44833, 44838, 44881, 44889, 44894, 44976, 44992, 45026, 45046, 45054, 45073, 45094, 45103, 45113, 45135, 45181, 45278, 45341, 45348, 45380, 45451, 45552, 45619, 45634, 45639, 45651, 45719, 45779, 45843, 45848, 45881, 45911, 45924, 46030, 46113, 46115, 46352, 46528, 46598, 46607, 46674, 46715, 46789, 46850, 46871, 46880, 47060, 47097, 47135, 47184, 47247, 47258, 47272, 47358, 47412, 47592, 47609, 47681, 47726, 47960, 48034, 48146, 48150, 48223, 48254, 48260, 48295, 48321, 48355, 48457, 48833, 48877, 48911, 48914, 48923, 48941, 48958, 48962, 49070, 49088, 49168, 49171, 49331, 49346, 49362, 49377, 49385, 49447, 49499, 49590, 49675, 49797, 49834, 49851, 49869, 49925, 49941, 49944, 50056, 50072, 50137, 50157, 50188, 50200, 50329, 50349, 50389, 50429, 50441, 50496, 50532, 50635, 50785, 50817, 50841, 50874, 50920, 50947, 50983, 51096, 51101, 51149, 51152, 51184, 51322, 51475, 51520, 51780, 51803, 51807, 51819, 51843, 51862, 51970, 52080, 52204, 52215, 52237, 52257, 52260, 52273, 52277, 52338, 52551, 52564, 52568, 52657, 52796, 52797, 52809, 52893, 52966, 53081, 53167, 53172, 53177, 53193, 53385, 53423, 53437, 53474, 53595, 53635, 53716, 53739, 53810, 53832, 53834, 53881, 53921, 53928, 54060, 54084, 54085, 54126, 54136, 54233, 54256, 54338, 54342, 54438, 54483, 54508, 54514, 54547, 54575, 54664, 54719, 54787, 54826, 54835, 54928, 54984, 55004, 55039, 55256, 55423, 55493, 55822, 56023, 56109, 56187, 56195, 56199, 56340, 56474, 56520, 56644, 56662, 56695, 56701, 56717, 56754, 56760, 56764, 56778, 56780, 56907, 56913, 56933, 56997, 57025, 57034, 57035, 57079, 57082, 57154, 57163, 57184, 57191, 57258, 57425, 57433, 57479]

answers = [['Id', 'Lat', 'Lon']]
for i in range(0,len(preds)):
    answers += [[IDs[i], preds[i][0], preds[i][1]]]

myFile = open('answers.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(answers)
