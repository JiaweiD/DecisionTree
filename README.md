# DecisionTree

The decision tree learning is to create a tree which has great generalization ability. Obviously, the key of learning process is how to choose an attribute to split the dataset.

We hope that the sample classification in the branches can be as little as possible, which means the purity can be as high as possible. However, information gain is a regular measurement of dataset purity.

A dataset(D) entrophy is defined as:
                  Ent(D) = -Σpk*log(2,pk)     k = 1,2,3...|y|
                  
When splitting the dataset by a certain attribute(a) which has several(|v|) possible values, the information gain is defined as:
                  Gain(D,a) = Ent(D) - ΣDv/D*Ent(Dv)     v = 1,2,3...|v|

So we calculate every attribute's information gain to choose the biggest one.
                  a* = argmax Gain(D,a)     a ∈ A

This kind of decision tree is called ID3 decision tree which has a preferrence of attributes of many possiblities of values. To avoid this affection, we can use information gain ratio or gini index instead of information gain.

The gain ratio is defined as :
                  Gain_ratio(D,a) = Gain(D,a)/IV(a)
                  IV(a) = -ΣDv/D*log(2,Dv/D)     v = 1,2,3...|v|
IV(a) is the intrinsic value of the attribute.

So we calculate every attribute's information gain ratio to choose the biggest one.
                  a* = argmax Gain_ratio(D,a)     a ∈ A

This kind of decision tree is called C4.5 decision tree which has a preferrence of attributes of small amount of possiblities of values. To avoid this affection, we can first choose attributes whose information gain is above average and then choose the one those gain ratio is the highest.

The gini value can also be used to measure dataset purity, which is defined as:
                  Gini(D) = ΣΣpk*pk' = 1 - Σpk^2     k = 1,2,3...|y|   k' ≠ k

The gini index is defined as:
                  Gini_index(D,a) = ΣDv/D*Gini(Dv)     v = 1,2,3...|v|

So we calculate every attribute's gini index to choose the smallest one.
                  a* = argmax Gini_index(D,a)     a ∈ A
                  
This kind of decision tree is called CART decision tree which can be used in both classification and regression.

The detail algorithm and code is shown in 'trees.py'.

So far, we can train datasets with discrete attributes to create a decision tree, but how to deal with continuous attributes(seeing 'Continuous value processing.txt'), how to process missing calues(seeing 'missing value processing.txt') and how to avoid over-fitting(seeing 'Pruning.txt') is so far need to be considered.
