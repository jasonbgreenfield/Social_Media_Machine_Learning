import numpy as np
import statistics as stat
import pickle as pkl
import time
# import matplotlib.pyplot as plt
import math
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

def createFeatures(i, X_te, X_tr, y_tr, usersFriends, indexToIDMatch):

    lat_list = []
    lon_list = []
    hours1 = []
    hours2 = []
    hours3 = []
    hours = []
    hour1_mean = X_te[i][1]
    hour2_mean = X_te[i][2]
    hour3_mean = X_te[i][3]
    friend_count = len(usersFriends[X_te[i][0]])
    friend_post_count = []
    mean_friend_post_count = 0
    for friend in usersFriends[X_te[i][0]]:
        x = indexToIDMatch[friend]
        if x != -1:
            if y_tr[x][0] != 0 and y_tr[x][1] != 0:
                lat_list += [y_tr[x][0]]
                lon_list += [y_tr[x][1]]
            friend_post_count += [int(X_tr[x][4])]
            if X_tr[x][3] != 25:
                hours3 += [int(X_tr[x][3])]
            if X_tr[x][2] != 25:
                hours2 += [int(X_tr[x][2])]
            if X_tr[x][1] != 25:
                hours1 += [int(X_tr[x][1])]

    if len(lat_list) > 0:
        lat_mean = stat.mean(lat_list)
        lon_mean = stat.mean(lon_list)
        # round_lat = []
        # round_lon = []
        # for i in range(0,len(lat_list)):
        #     val_lat = math.floor(lat_list[i]/10)
        #     rem_lat = math.floor(lat_list[i])%10
        #     #print(f"{lat_list[i]} - {val_lat} - {rem_lat}")
        #     val_lon = math.floor(lon_list[i]/10)
        #     rem_lon = math.floor(lon_list[i])%10
        #     if rem_lat > 5:
        #         round_lat += [val_lat*10 + 10]
        #     else:
        #         round_lat += [val_lat*10]
        #     if rem_lon > 5:
        #         round_lon += [val_lon*10 + 10]
        #     else:
        #         round_lon += [val_lon*10]
        # lat_mode = mode(round_lat)
        # lon_mode = mode(round_lon)
        #
        # short_list_lat = []
        # short_list_lon = []
        # for i in range(0,len(lat_list)):
        #     if round_lat[i] == lat_mode:
        #         short_list_lat += [lat_list[i]]
        #     if round_lon[i] == lon_mode:
        #         short_list_lon += [lon_list[i]]
        #
        # best_guess_lat = stat.median(short_list_lat)
        # best_guess_lon = stat.median(short_list_lon)

    else:
        # best_guess_lat = 40
        # best_guess_lon = -80
        lat_mean = 40
        lon_mean = -80



    if len(hours1) > 0:
        """ Here we may want to only look at most posting friends hours only."""
        hour1_mean = round(stat.mean(hours1),2)

    if len(hours2) > 0:
        hour2_mean = round(stat.mean(hours2),2)

    if len(hours3) > 0:
        hour3_mean = round(stat.mean(hours3),2)

    if len(friend_post_count) > 0:
        mean_friend_post_count = round(stat.mean(friend_post_count),2)

    return friend_count, mean_friend_post_count, lat_mean, lon_mean, hour1_mean, hour2_mean, hour3_mean

# def mode(list):
#     count = Counter(list)
#     max = 0
#     max_list = []
#     for k,v in count.items():
#         if v >= max:
#             max = v
#     for k,v in count.items():
#         if v == max:
#             max_list += [k]
#     return max_list[0]

# def standardize(X):
#     mins = []
#     maxs = []
#     for i in range(len(X[0])):
#         temp_row = []
#         for j in range(len(X)):
#             temp_row.append(X[j][i])
#         temp_row.sort()
#         mins.append(temp_row[0])
#         maxs.append(temp_row[len(temp_row)-1])
#
#     for i in range(len(X)):
#         for j in range(len(X[0])):
#             X[i][j] = (X[i][j] - mins[j]) / (mins[j] - maxs[j])
#     return -1*X


if __name__ == "__main__":
    f_te = open("Data/posts_test.txt", "r")
    f_tr = open("Data/posts_train.txt", "r")
    friend_file = open("Data/graph.txt", "r")
    f1_te = f_te.readlines()
    f1_tr = f_tr.readlines()
    friends = friend_file.readlines()

    for post in range(0,len(f1_te)):
        f1_te[post] = f1_te[post][0:-1]
        f1_te[post] = f1_te[post].split(',')

    for post in range(0,len(f1_tr)):
        f1_tr[post] = f1_tr[post][0:-1]
        f1_tr[post] = f1_tr[post].split(',')

    # This is our test set that did not come with lat and lon
    f1_te = np.asarray(f1_te)
    f1_te = np.delete(f1_te,0,0)

    # This is all of our data that came with lat and lon
    f1_tr = np.asarray(f1_tr)
    f1_tr = np.delete(f1_tr,0,0)

    # Split training set into X and y
    y_tr = f1_tr[:, [4,5]]
    X_tr = f1_tr[:, [0,1,2,3,6]]

    X_tr = X_tr.astype(int)
    y_tr = y_tr.astype(float)

    X_te = f1_te.astype(int)

    for fri in range(0,len(friends)):
        friends[fri] = friends[fri][0:-1]
        friends[fri] = friends[fri].split('\t')
    friends = np.asarray(friends)
    friends = friends.astype(int)

    # maps UID to index in X_tr
    indexToIDMatch = np.zeros(57563)
    for i in range(0,len(indexToIDMatch)):
        indexToIDMatch[i] = -1

    for i in range(0,len(X_tr)):
        indexToIDMatch[X_tr[i][0]] = i
    indexToIDMatch = indexToIDMatch.astype(int)

    # maps UID to index in X_te
    indexToIDMatch_te = np.zeros(57563)
    for i in range(0,len(indexToIDMatch_te)):
        indexToIDMatch_te[i] = -1

    for i in range(0,len(X_te)):
        indexToIDMatch_te[X_te[i][0]] = i
    indexToIDMatch_te = indexToIDMatch_te.astype(int)


    usersFriends = [[] for i in range(57563)]
    usersFriends[0] = None
    for i in range(0,len(friends)):
        usersFriends[friends[i][0]] += [friends[i][1]]
    # usersFriends is a list of lists. Each nested list is a list of each person friends

    new_X_te = [None]*len(X_te)
    for i in range(0, len(X_te)):
        a = np.concatenate((X_te[i], createFeatures(i, X_te, X_tr, y_tr, usersFriends, indexToIDMatch)),axis=None)
        new_X_te[i] = a
        new_X_te[i] = np.asarray(new_X_te[i])

    X_te = np.asarray(new_X_te)

    tempX = [None]*(len(X_te))
    for i in range(0,len(X_te)):
        tempX[i] = X_te[i][1:]
    X_te = tempX

    scaler = MinMaxScaler()
    scaler.fit(X_te)
    X_te = scaler.transform(X_te)

    desc = "Design Matrix for our synthesized data test set."
    data = X_te
    pickle_out = open('./Data/test_final_standardized.pkl', 'wb')
    pkl.dump((data, desc), pickle_out)
    pickle_out.close()


    # # END
