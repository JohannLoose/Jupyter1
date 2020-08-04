import wzalgorithm as wz
from wzalgorithm import REvolSuccessPredicate
import NeuralNets as nn

class myEvaluator:
    def __init__(self, targetNodeNR, skeleton, inputSet, targetSet):
        self.targetNodeNR = targetNodeNR
        self.skeleton = skeleton
        self.inputSet = inputSet
        self.targetSet = targetSet

    def evaluate(targetNodeNR, skeleton, inputSet, targetSet, biasesAndWeights):
        return nn.wz_evaluateFitness(targetNodeNR, skeleton, inputSet, targetSet, biasesAndWeights)


    def __call__(self, individual):
        individual.restrictions.resize(1)
        individual.restrictions[0] = myEvaluator.evaluate(self.targetNodeNR, self.skeleton, self.inputSet, self.targetSet, individual.parameters)
        return individual.restrictions[0] < 0.01


def performRevol(targetNodeNR, skeleton, inputSet, targetSet):
    revol = wz.REvol()
    origin = revol.generateOrigin(len(skeleton.connections) + len(skeleton.nodes))
    revol.maxEpochs(10000)
    result = revol.run(origin, wz.REvolSuccessPredicate(myEvaluator(targetNodeNR, skeleton, inputSet, targetSet)))

    i=0
    for x in skeleton.nodes:
        skeleton.nodes[x].bias = result.parameters[i]
        i+=1
    for x in skeleton.connections:
        skeleton.connections[x].weight = result.parameters[i]
        i+=1

    skeleton.fitness = nn.wz_evaluateFitness(targetNodeNR, skeleton, inputSet, targetSet, result.parameters)
    return skeleton


