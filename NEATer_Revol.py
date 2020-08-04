import wzalgorithm as wz
from wzalgorithm import REvolSuccessPredicate
import Neat
import ProblemSets
from veithRevol import performRevol
import NeuralNets as nn

problemSet = ProblemSets.XorSet()
inputSet = problemSet.inputSet
targetSet = problemSet.targetSet
startingIndividual = problemSet.startingIndividual


neat = Neat.NEAT(inputSet, targetSet, startingIndividual)
result = neat.basicNeat()
print("NEAT abgeschlossen, Ergebnis: " + str(result.fitness))
for x in result.connections:
    connection = result.connections[x]
    if connection.enabled == True:
        print("Verbindung %i von %i -> %i: %f" %(connection.innovationNR, connection.start, connection.end, connection.weight))
for x in result.nodes:
    node = result.nodes[x]
    print("Knoten Nr. %i hat Bias %f" % (node.nodeNR, node.bias))
k=1


