import numpy as np

class inferenceGenerator:
	def __init__(self, model, best_model_path, Xtest, ytest,threshold):
		self.model=model
		self.best_model_path=best_model_path
		self.model.load_weights(self.best_model_path)
		self.Xtest=Xtest
		self.ytest = ytest
		self.predictions=self.predict(self.Xtest)
		self.threshold=threshold

	def predict(self,testset):
		return self.model.predict(testset)

	def model_output_consumer(self):
	#     print(predictions.shape)
	    flattened_pred=np.array([x.ravel() for x in self.predictions])
	#     print(flattened_pred.shape)
	    results=[]
	    for pred in flattened_pred:
	        result = [0 if x<self.threshold else 1 for x in pred]
	        results.append(result)
	    return np.array(results)


	