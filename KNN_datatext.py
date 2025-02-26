import numpy as np
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
class KNN:
	def __init__(self, k : int, dataset : np.array) -> None:
		self.k = k
		self.dataset = dataset

	def counts_value(self) -> dict:
		data = self.dataset[: self.k]	
		array_set = list(set([int(data[i][1]) for i in range(len(data))]))
		array_counts = dict([[array_set[i],0] for i in range(len(array_set))])
		for i in range(len(data)):
			array_counts[data[i][1]] += 1
		return array_counts
	
	def selection_value_max(self) -> int:
		value_max = -1
		for key, value in self.counts_value().items():
			value_max = max(value_max,value)

		label = -1
		for key, value in self.counts_value().items():
			if (value_max == value):
				label = key
		return label

def distance(p1 : np.array, p2 : np.array) -> int:
	total = 0
	for i in range(len(p1)):
		total += abs(p1[i] - p2[i]) ** 2
	return sqrt(total)

def get_data() -> list:	
	data = ["góp gió gặt bão","có làm mới có ăn","đất lành chim đậu","ăn cháo đá bát","gậy ông đập lưng ông","qua cầu rút ván"]
	return data

def binary_search(arr : list, char : str) -> bool:
	l = 0
	r = len(arr) - 1
	ans = -1
	while(l <= r):
		mid = (l + r) // 2
		if (arr[mid] == char):
			return mid
		elif (arr[mid] < char):
			l = mid + 1
		else:
			r = mid - 1
	return ans

def format_dataset_test(char_array : list) -> np.array:
	char_array = np.array(char_array)
	s = 'không làm cạp đất mà ăn'
	data = s.split()
	data_new = [0] * 23
	cnt = 0
	for i in data:
		index = binary_search(char_array,i)
		if (index != -1):
			data_new[index] += 1
		cnt += 1
	return np.array(data_new)

def init_model(x_train : np.array, y_train : np.array, x_test : np.array, k : int) -> int:
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(x_train,y_train)
	return model.predict(x_test)[0]

def format_dataset_ID(data : list[str], labels : np.array, k : int) -> list[int]:
	data = [data[i].split() for i in range(len(data))]
	char_array = []
	for i in range(len(data)):
		for j in range(len(data[i])):
			char_array.append(data[i][j])

	char_array = list(set(char_array))
	char_array.sort()

	data_new = []
	for i in range(len(data)):
		data_tmp = []
		for j in range(len(char_array)):
			data_tmp.append(0)
		data_new.append(data_tmp)

	for i in range(len(data)):
		for j in range(len(data[i])):
			index = binary_search(char_array,data[i][j])
			if (index != -1):
				data_new[i][index] += 1

	data_new = np.array(data_new)

	x_test = [format_dataset_test(char_array)]
	distance_and_labels = [[distance(x_test[0],data_new[i]), int(labels[i])] for i in range(len(data_new))]
	distance_and_labels.sort()
	x_test = np.array(x_test)
	model_KNN = KNN(k,distance_and_labels)

	print("label using code in rice = ", model_KNN.selection_value_max())
	print("label using model = ", init_model(data_new,labels,x_test,k))
	

def get_labels(data : list[str]) -> np.array:
	return np.array([1,1,1,0,0,0])

def main():
	labels = get_labels(get_data())	
	k = 3
	format_dataset_ID(get_data(), labels, k)
main()





