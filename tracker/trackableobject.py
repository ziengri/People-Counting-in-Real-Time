from collections import deque

class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		# self.centroids = [centroid]
		# Храним только последние 50 центроидов
		self.centroids = deque([centroid], maxlen=50)

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False