import trees as tr
import treePlotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = tr.createTree(lenses,lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)
