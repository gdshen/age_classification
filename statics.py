import numpy as np

a = np.load('predicted_and_real3.npy')
# print(a[:10, :])
# a.sort(axis=0)
a = a[a[:, 1].argsort()]
for k in range(1, 10):
    rangeMax = k * 10
    num = 0
    sum = 0
    for i in a:
        if i[1] < rangeMax and i[1] > rangeMax - 9:
            num = num + 1
            sum = sum + abs(i[0] - i[1])
    if sum != 0:
        print('%d到%d的样本总数为%d，平均误差为%f' % (rangeMax - 9, rangeMax, num, sum / num))

for i in a:
    if i[1] < 21:
        print(i)

# a = a[:10, :]
print(np.mean(np.abs(a[:, 0] - a[:, 1])))
