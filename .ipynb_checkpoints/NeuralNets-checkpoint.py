import math
import numpy as np

class Connection:
    def __init__(self, start, end, weight, innovationNR):
        self.start = start
        self.end = end
        self.weight = weight
        self.innovationNR = innovationNR
        self.enabled = True
        self.scatter = 0.1

class Node:
    isReady = False
    scatter = 0.1
    inputSum = 0

    def __init__(self, nodeNR, type):
        self.nodeNR = nodeNR
        self.type = type
        self.bias = 0
        self.predecessorConnections = []

    def refresh(self):
        self.inputSum = 0
        self.isReady = False

    def addPredecessorConnection(self, connection):
        self.predecessorConnections.append(connection)

    def removePredecessorConnection(self, innovationNR):
        for i in range (0, len(self.predecessorConnections)):
            if self.predecessorConnections[i].innovationNR == innovationNR:
                self.predecessorConnections.pop(i)
                break


class Individual:
    fitness = -42

    def __init__(self):
        self.connections = {}
        self.nodes = {}
        self.ttl = -1

    def addConnection(self, connection):
        self.connections[connection.innovationNR] = connection

    def addNode(self, node):
        self.nodes[node.nodeNR] = node

    def setTtl(self, t):
        self.ttl = t

    def age(self):
        self.ttl -=1
        if self.ttl < 0:
            self.fitness = -1

    def refreshNodes(self):
        for x in self.nodes:
            self.nodes[x].refresh()

def safeSigmoid(x):
    x = max(x, -100)
    return 1/(1+math.exp(-x))

def identity(x):
    return x
    

def recOutput(targetNodeNR, individual, inputValues):
    activation = safeSigmoid
    targetNode = individual.nodes[targetNodeNR]

    #assign sum of inputs to node, mark it as "ready"
    if not targetNode.isReady:
        if targetNode.type == "input":
            targetNode.inputSum = inputValues[targetNodeNR]
        else:
            for connection in targetNode.predecessorConnections:
                if connection.enabled:
                    targetNode.inputSum += recOutput(connection.start, individual, inputValues) * connection.weight
        targetNode.isReady = True
    
    #return either input values or inputsums + bias
    if targetNode.type == "input":
        return targetNode.inputSum
    else:
        return activation(targetNode.inputSum + targetNode.bias)


def evaluateFitness(targetNodeNR, individual, inputSet, outputSet): #this assumes that the net has only one output
    fitness = len(outputSet)
    for i in range(0, len(inputSet)):
        fitness -= abs((recOutput(targetNodeNR, individual, inputSet[i]) - outputSet[i]))
        individual.refreshNodes()
    return fitness


def wz_evaluateFitness(targetNodeNR, skeleton, inputSet, targetSet, biasesAndWeights): #WeightsBiases
    i=0
    for x in skeleton.nodes:
        skeleton.nodes[x].bias = biasesAndWeights[i]
        i+=1
    for x in skeleton.connections:
        skeleton.connections[x].weight = biasesAndWeights[i]
        i+=1

    fitness = 0 
    for i in range(0, len(inputSet)):
        fitness += abs((recOutput(targetNodeNR, skeleton, inputSet[i]) - targetSet[i]))
        skeleton.refreshNodes()

    return fitness #this is more of an error value
