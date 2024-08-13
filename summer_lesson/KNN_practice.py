import numpy as np
np.set_printoptions(threshold=np.inf)  ## print all values of matrix without reduction
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance     ## calculate the distance between two points


## first part : load dataset(very famous called iris)
"""
your code
"""
iris = datasets.load_iris()
iris_data = iris.data
iris_label =iris.target

#print(iris_data)
#print(iris_label)

## second part : choose the label that we want (can be based on your preference)
"""
your code
"""
column = [2,3]
iris_data = iris_data[:,column]
print(iris_data)
## third part : plot the distribution of data
"""
your code
"""
for i in range(iris_label.shape[0]):

    if iris_label[i] == 0:
        plt.scatter(iris_data[i,0],iris_data[i,1],color='red',s=50,alpha=0.6)

    if iris_label[i] == 1:
        plt.scatter(iris_data[i,0],iris_data[i,1],color='green',s=50,alpha=0.6)

    if iris_label[i] == 2:
        plt.scatter(iris_data[i,0],iris_data[i,1],color='blue',s=50,alpha=0.6)
# plt.show()
## forth part : the principle of KNN
"""
your code 

1. determine the value of K, the category of data, and the test point
2. calculate the distance between the test point and all data
3. find the top k nearest data
4. select the categories with the most votes
"""
K = 5
class_num = 3
class_count = [0,0,0]
test_point = [3,2]
dis_array = []
for i in range(iris_label.shape[0]):
    dst = distance.euclidean(test_point, iris_data[i, :])
    dis_array.append(dst)
    
idx_sort = np.argsort(dis_array)[0:K]

for i in range(K):
    label = iris_label[idx_sort[i]]
    class_count[label] += 1
    
print(class_count)
result = np.argsort(class_count)[-1]
# print(result)

## fifth part : plot the test point

"""
your code
"""
if result == 0:
    plt.scatter(test_point[0], test_point[1], color="red", s=150, alpha=1, marker="^")
elif result == 1:
    plt.scatter(test_point[0], test_point[1], color="green", s=150, alpha=1, marker="^")
elif result == 2:
    plt.scatter(test_point[0], test_point[1], color="blue", s=150, alpha=1, marker="^")
        
plt.show()