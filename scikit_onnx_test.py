from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy

iris = load_iris() 
X, y = iris.data, iris.target 
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LogisticRegression()
clr.fit(X_train, y_train)

print("First X row:")
print(X[:1])

initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx = convert_sklearn(clr, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onnx.SerializeToString())

sess = rt.InferenceSession("logreg_iris.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onnx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
print("y:")
print(y_test)
print("pred_onnx:")
print(pred_onnx)