from sklearn import tree
from sklearn.externals.six import StringIO
import pandas as pd
import numpy as np
import pydotplus

data = pd.read_table('/Users/ChrisD/PycharmProjects/MLstudying/trees/lenses.txt',header = None,delim_whitespace = True)
dataSet= data.iloc[:,[0,1,2,3]]
labelSet = data.iloc[:,4]
dataConvertList={}
labelConvertList={}

#convert string attribute values to integers
for feat in np.array(dataSet).transpose():
    i = 1
    for data in feat:
        if data not in dataConvertList.keys():
            dataConvertList[data]=i
            i=i+1     
i=1
for data in labelSet:
    if data not in labelConvertList.keys():
        labelConvertList[data]=i
        i=i+1
for key in dataConvertList:
    dataSet = dataSet.replace(key,dataConvertList[key])
for key in labelConvertList:
    labelSet = labelSet.replace(key,labelConvertList[key])
    
#train a model
model = tree.DecisionTreeClassifier(criterion='entropy',random_state = 0)
s = model.fit(dataSet,labelSet)

#plot the decision tree by pydotplus
tree_file = StringIO()
tree.export_graphviz(model,out_file = tree_file)
graph = pydotplus.graph_from_dot_data(tree_file.getvalue())
graph.write_pdf("tree.pdf")
