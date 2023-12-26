def computeMAE(list_a, list_b):
    MAE_temp = []
    for num in range(len(list_a)):
        MAE_temp.append(abs(list_a[num] - list_b[num]))
    MAE = sum(MAE_temp) / len(list_a)
    return MAE


def computeMSE(list_a, list_b):
    MSE_temp = []
    for num in range(len(list_a)):
        MSE_temp.append((list_a[num] - list_b[num]) * (list_a[num] - list_b[num]))
    MSE = sum(MSE_temp) / len(list_a)
    return MSE
