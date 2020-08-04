import numpy as np
import copy
import NeuralNets as nn
from veithRevol import performRevol
import random
import matplotlib.pyplot as plt

def unNest(nestedList):
    longList = []
    for list in nestedList:
        longList += list
    return longList

def measureGeneticDistance(connectionSet_1, connectionSet_2, parameters): #instead of Disjoint and Excess i'll use Missmatched genes M=D+E
    c1, c2, c3 = parameters[0], parameters[1], parameters[2]

    W=0
    M=0
    matchingGenesCounter = 0
    for x in connectionSet_1:
        if x in connectionSet_2 : 
            W+= abs(connectionSet_1[x].weight * connectionSet_1[x].enabled - connectionSet_2[x].weight)
            matchingGenesCounter += 1
        else: M+=1
    for y in connectionSet_2:
        if y not in connectionSet_1: M+=1
    
    if matchingGenesCounter == 0: result = c1*M
    else: result = c1*M + c3*(W/matchingGenesCounter)
    return result

def speciate(population, distance, parameters):
    p0 = population.pop(0)
    speciatedList = [[p0]]
    for individual in population:
        matchingSpeciesFound = False
        for species in speciatedList:
            if measureGeneticDistance(individual.connections, species[0].connections, parameters) <= distance:
                species.append(individual)
                matchingSpeciesFound = True
                break
        if not matchingSpeciesFound:
            speciatedList.append([individual])
    population.append(p0)
    return speciatedList

def crossover(ind1, ind2):
    #make sure ind1 has better fitness-score
    if ind1.fitness < ind2.fitness:
        ind1a = ind1
        ind2a = ind2

        ind1, ind2 = ind2, ind1

    newIndividual = nn.Individual()
    #treat the bias of each node like a connection from a hypothetical bias-node. 
    #so if the node exists in both parents, choose 50/50, else only take from best parent
    for x in ind1.nodes:
        if x in ind2.nodes and random.choice([0, 1]) == 0:
            newIndividual.addNode(copy.deepcopy(ind2.nodes[x]))

        else:
            newIndividual.addNode(copy.deepcopy(ind1.nodes[x]))

    for x in ind1.connections:
        if x in ind2.connections and random.choice([0, 1]) == 0:
            newIndividual.addConnection(copy.deepcopy(ind2.connections[x]))
        else:
            newIndividual.addConnection(copy.deepcopy(ind1.connections[x]))

    #for each node drop predecessor connections that aren't in the genome.
    #also get the proper connection if it is in the genome
    for x in newIndividual.nodes:
        node = newIndividual.nodes[x]
        for i in range(len(node.predecessorConnections)-1, -1, -1):
            iNR = node.predecessorConnections[i].innovationNR
            if iNR in newIndividual.connections:
                node.predecessorConnections[i] = newIndividual.connections[iNR]
            else:
                node.predecessorConnections.pop(i)
    return newIndividual
    
def checkLegality(individual, start, end):
    predecessors = []
    individual.writeIndividualToFile()
    def recursivePredecessors(node, predecessors):
        for predecessorConnection in node.predecessorConnections:
            predecessors.append(predecessorConnection.start)
            recursivePredecessors(individual.nodes[predecessorConnection.start], predecessors)
    
    recursivePredecessors(individual.nodes[start], predecessors)
    if end in predecessors: return False
    else: return True

def addNode(individual, innovationLog):
    connectionToSplit = individual.connections[np.random.choice(list(individual.connections.keys()))]

    if connectionToSplit.enabled == False:
        return None

    #check if this connection has been split before
    if innovationLog.splitTable[connectionToSplit.start][connectionToSplit.end] == -1: #connection has NOT been split before
        innovationLog.splitTable[connectionToSplit.start][connectionToSplit.end] = innovationLog.nodeCounter
        newNode = nn.Node(innovationLog.nodeCounter, "hidden")
        innovationLog.nodeCounter+=1

        newConnection1 = nn.Connection(connectionToSplit.start, newNode.nodeNR, 1, innovationLog.innovationCounter)
        innovationLog.innovationTable[connectionToSplit.start][newNode.nodeNR] = innovationLog.innovationCounter
        innovationLog.innovationCounter+=1

        newConnection2 = nn.Connection(newNode.nodeNR, connectionToSplit.end, connectionToSplit.weight, innovationLog.innovationCounter)
        innovationLog.innovationTable[newNode.nodeNR][connectionToSplit.end] = innovationLog.innovationCounter
        innovationLog.innovationCounter += 1
    else: #connection HAS been split before
        newNode = nn.Node(innovationLog.splitTable[connectionToSplit.start][connectionToSplit.end], "hidden")
        newConnection1=nn.Connection(connectionToSplit.start, newNode.nodeNR, 1, innovationLog.innovationTable[connectionToSplit.start][newNode.nodeNR])
        newConnection2=nn.Connection(newNode.nodeNR, connectionToSplit.end, connectionToSplit.weight, innovationLog.innovationTable[newNode.nodeNR][connectionToSplit.end])

    individual.nodes[connectionToSplit.end].removePredecessorConnection(connectionToSplit.innovationNR)
    individual.connections[connectionToSplit.innovationNR].enabled = False
    del individual.connections[connectionToSplit.innovationNR]

    newNode.addPredecessorConnection(newConnection1)
    individual.nodes[connectionToSplit.end].addPredecessorConnection(newConnection2)

    individual.addNode(newNode)
    individual.addConnection(newConnection1)
    individual.addConnection(newConnection2)
    l=0

def createNewConnection(individual, innovationLog):
    """Chooses two nodes at RANDOM and checks, if this connection would be legal.
    If it is, it's added to the individual and the connection's innovationNR is either added to or taken from the innovationTable"""
    nodeKeys = list(individual.nodes.keys())
    newStart = individual.nodes[np.random.choice(nodeKeys)].nodeNR
    newEnd = individual.nodes[np.random.choice(nodeKeys)].nodeNR

    if (not(newStart == newEnd)) and checkLegality(individual, newStart, newEnd) and individual.nodes[newEnd].type != "input":
        if innovationLog.innovationTable[newStart][newEnd]==-1:
            innovationLog.innovationTable[newStart][newEnd] = innovationLog.innovationCounter
            innovationNR = innovationLog.innovationCounter
            innovationLog.innovationCounter+=1
        else: innovationNR = innovationLog.innovationTable[newStart][newEnd]
        
        if innovationNR in individual.connections:
            individual.connections[innovationNR].enabled = True
        else:
            newConnection = nn.Connection(newStart, newEnd, np.random.normal(0,1), innovationNR)
            individual.addConnection(newConnection)
            individual.nodes[newEnd].addPredecessorConnection(newConnection)

def switchConnection(individual):
    connection = individual.connections[np.random.choice(list(individual.connections.keys()))]
    connection.enabled = not connection.enabled

def mutateWeights(individual):
    for x in individual.connections:
        individual.connections[x].weight += np.random.normal(0,0.4)

    for y in individual.nodes:
        individual.nodes[y].bias += np.random.normal(0,0.4)

def mutate(individual, innovationLog, newNodeChance, newConnectionChance, mutateWeightsChance, switchConnectionChance):
    newIndividual = copy.deepcopy(individual)

    
    if np.random.uniform(0,1) < newNodeChance:
        addNode(newIndividual, innovationLog)
    if np.random.uniform(0,1) < newConnectionChance:
        createNewConnection(newIndividual, innovationLog)
    if np.random.uniform(0,1) < mutateWeightsChance:
        mutateWeights(newIndividual)
    #if np.random.uniform(0,1) < switchConnectionsChance:
    #    switchConnection(individual)
    return newIndividual



def reproduceProportional(speciatedList, innovationLog, newNodeChance, newConnectionChance, mutateWeightsChance, switchConnectionChance):
    population = []
    speciesFitnessValues=[None]*len(speciatedList)

    #get the average fitness of each species
    for i in range(0, len(speciatedList)):
        speciesFitSum = 0
        for individual in speciatedList[i]:
            speciesFitSum += individual.fitness**2
        speciesFitnessValues[i] = speciesFitSum / len(speciatedList[i])
    totalFitSum = sum(speciesFitnessValues)

    popSize = 100
    for k in range(0, len(speciatedList)):
        if len(speciatedList[k])==1:
            for i in range(0, round(speciesFitnessValues[k] / totalFitSum * popSize)): 
                population.append(mutate(speciatedList[k][0], innovationLog, newNodeChance, newConnectionChance, mutateWeightsChance, switchConnectionChance))
        else:
            for i in range(0, round(speciesFitnessValues[k] / totalFitSum * popSize)): 
                s = random.sample(speciatedList[k],2)
                ind1, ind2 = s[0], s[1]
                newIndividual = crossover(ind1, ind2)
                population.append(mutate(newIndividual, innovationLog, newNodeChance, newConnectionChance, mutateWeightsChance, switchConnectionChance))
    #TODO? fix populationNumbers
    return population

def cataclysm(speciatedList):
    newSpeciatedList =[]
    bestSpeciesNumber = 0
    for k in range(0,2):
        bestFitness = 0
        for i in range(0,len(speciatedList)):
            if speciatedList[i][0].fitness > bestFitness:
                bestSpeciesNumber = i
                bestFitness = speciatedList[i][0].fitness
        newSpeciatedList.append(speciatedList.pop(bestSpeciesNumber))
    return newSpeciatedList
        
def printStatistics(speciatedList):
    speciesCount = len(speciatedList)
    bestIndividual = speciatedList[0][0]
    worstIndividual = speciatedList[0][0]
    weightBiasSum = 0
    connectionCount = 0
    nodeCount = 0
    fitnessSum = 0

    for species in speciatedList:
        individual = species[0]
        if individual.fitness > bestIndividual.fitness:
            bestIndividual = individual
        elif individual.fitness < worstIndividual.fitness:
            worstIndividual = individual
        for x in individual.connections:
            weightBiasSum += abs(individual.connections[x].weight)
            connectionCount +=1
        for x in individual.nodes:
            if individual.nodes[x].type != "input":
                weightBiasSum += abs(individual.nodes[x].bias)
            nodeCount += 1
        fitnessSum += individual.fitness

    avgWeightBiasSum = weightBiasSum / speciesCount
    avgConnectionCount = connectionCount / speciesCount
    avgNodeCount = nodeCount / speciesCount
    avgFitness = fitnessSum / speciesCount

    bestWeightBiasSum = 0
    for x in bestIndividual.connections:
        bestWeightBiasSum += abs(bestIndividual.connections[x].weight)
    for x in bestIndividual.nodes:
        if bestIndividual.nodes[x].type != "input":
            bestWeightBiasSum += abs(bestIndividual.nodes[x].bias)
    bestConnectionCount = len(bestIndividual.connections)
    bestNodeCount = len(bestIndividual.nodes)
    bestFitness = bestIndividual.fitness

    worstWeightBiasSum = 0
    for x in worstIndividual.connections:
        worstWeightBiasSum += abs(worstIndividual.connections[x].weight)
    for x in worstIndividual.nodes:
        if worstIndividual.nodes[x].type != "input":
            worstWeightBiasSum += abs(worstIndividual.nodes[x].bias)
    worstConnectionCount = len(worstIndividual.connections)
    worstNodeCount = len(worstIndividual.nodes)
    worstFitness = worstIndividual.fitness

    print("%f; %f; %f; %f; %f; %f; %f; %f; %f; %f; %f; %f; %f" %(speciesCount, avgWeightBiasSum, avgConnectionCount, avgNodeCount, avgFitness,
                                                                bestWeightBiasSum, bestConnectionCount, bestNodeCount, bestFitness, 
                                                                worstWeightBiasSum, worstConnectionCount, worstNodeCount, worstFitness))

class StatisticLog:
    def __init__(self):
        self.generationList=[]
        self.speciesCountList=[]
        self.avgWeightBiasSumList=[]
        self.avgConnectionCountList=[]
        self.avgNodeCountList=[]
        self.avgFitnessList=[]
        self.bestWeightBiasSumList=[]
        self.bestConnectionCountList=[]
        self.bestNodeCountList=[]
        self.bestFitnessList=[]
        self.worstWeightBiasSumList=[]
        self.worstConnectionCountList=[]
        self.worstNodeCountList=[]
        self.worstFitnessList=[]
    
    def update(self, speciatedList):
        speciesCount = len(speciatedList)
        bestIndividual = speciatedList[0][0]
        worstIndividual = speciatedList[0][0]
        weightBiasSum = 0
        connectionCount = 0
        nodeCount = 0
        fitnessSum = 0

        for species in speciatedList:
            individual = species[0]
            if individual.fitness > bestIndividual.fitness:
                bestIndividual = individual
            elif individual.fitness < worstIndividual.fitness:
                worstIndividual = individual
            for x in individual.connections:
                weightBiasSum += abs(individual.connections[x].weight)
                connectionCount +=1
            for x in individual.nodes:
                if individual.nodes[x].type != "input":
                    weightBiasSum += abs(individual.nodes[x].bias)
                nodeCount += 1
            fitnessSum += individual.fitness

        avgWeightBiasSum = weightBiasSum / speciesCount
        avgConnectionCount = connectionCount / speciesCount
        avgNodeCount = nodeCount / speciesCount
        avgFitness = fitnessSum / speciesCount

        bestWeightBiasSum = 0
        for x in bestIndividual.connections:
            bestWeightBiasSum += abs(bestIndividual.connections[x].weight)
        for x in bestIndividual.nodes:
            if bestIndividual.nodes[x].type != "input":
                bestWeightBiasSum += abs(bestIndividual.nodes[x].bias)
        bestConnectionCount = len(bestIndividual.connections)
        bestNodeCount = len(bestIndividual.nodes)
        bestFitness = bestIndividual.fitness

        worstWeightBiasSum = 0
        for x in worstIndividual.connections:
            worstWeightBiasSum += abs(worstIndividual.connections[x].weight)
        for x in worstIndividual.nodes:
            if worstIndividual.nodes[x].type != "input":
                worstWeightBiasSum += abs(worstIndividual.nodes[x].bias)
        worstConnectionCount = len(worstIndividual.connections)
        worstNodeCount = len(worstIndividual.nodes)
        worstFitness = worstIndividual.fitness

        self.generationList.append(len(self.generationList))
        self.speciesCountList.append(speciesCount)
        self.avgWeightBiasSumList.append( avgWeightBiasSum)
        self.avgConnectionCountList.append( avgConnectionCount)
        self.avgNodeCountList.append( avgNodeCount)
        self.avgFitnessList.append( avgFitness)
        self.bestWeightBiasSumList.append( bestWeightBiasSum)
        self.bestConnectionCountList.append( bestConnectionCount)
        self.bestNodeCountList.append( bestNodeCount)
        self.bestFitnessList.append( bestFitness)
        self.worstWeightBiasSumList.append( worstWeightBiasSum)
        self.worstConnectionCountList.append( worstConnectionCount)
        self.worstNodeCountList.append( worstNodeCount)
        self.worstFitnessList.append( worstFitness)




class InnovationLog:
    def __init__(self):
        self.innovationTable = np.full([1000,1000],-1)
        self.splitTable = np.full([1000,1000],-1)
        self.innovationCounter = 0
        self.nodeCounter = 0

class NEAT:
    def __init__(self, inputSet, targetSet, startingIndividual):
        self.innovationLog = InnovationLog()
        self.inputSet = inputSet
        self.targetSet = targetSet
        self.startingIndividual = startingIndividual
        self.initializeInnovationTable(self.startingIndividual)

    def initializeInnovationTable(self, individual):
        for x in individual.connections:
            self.innovationLog.innovationTable[individual.connections[x].start][individual.connections[x].end] = individual.connections[x].innovationNR
            self.innovationLog.innovationCounter += 1
        self.innovationLog.nodeCounter = len(individual.nodes)


    def neaterRevol(self):
        statisticLog = StatisticLog()
        startingIndividual = self.startingIndividual

        #initialize constants
        generationCounter = 0
        noSuccessCounter = 0
        bestFitness = 0
        maximumFitness = len(self.targetSet)
        targetError = 1e-1
        targetFitness = maximumFitness - targetError

        #get outputNode TODO: adapt for multiple nodes // iteriert über die dictionary-keys (nodeNRs) und bleibt beim passenden stehen
        for targetNodeNR in startingIndividual.nodes:
            if startingIndividual.nodes[targetNodeNR].type == "output":
                break

        startingIndividual.fitness = nn.evaluateFitness(targetNodeNR, startingIndividual, self.inputSet, self.targetSet)
        
        #initialize population
        population = []
        for i in range(0,100):
            population.append(copy.deepcopy(startingIndividual))
        
        print("starting NEATer REvol...")
        #print("speciesCount; avgWeightBiasSum; avgConnectionCount; avgNodeCount; avgFitness; bestWeightBiasSum; bestConnectionCount; bestNodeCount; bestFitness; worstWeightBiasSum; worstConnectionCount; worstNodeCount; worstFitness")
        while True:   
            #sort individuals into species
            parameters = (10,10,0)
            distance = 1
            speciatedList = speciate(population, distance, parameters)

            #for each species find optimized set of weights
            for i in range(0, len(speciatedList)):
                optimizedSpecimen = performRevol(targetNodeNR, speciatedList[i][0], self.inputSet, self.targetSet)
                optimizedSpecimen.fitness = maximumFitness - optimizedSpecimen.fitness
                #print("Optimized Species %i of %i in generation %i. Best Fitness is %f" %(i, len(speciatedList)-1, generationCounter, optimizedSpecimen.fitness))
                if optimizedSpecimen.fitness > targetFitness: break
                speciatedList[i] = [optimizedSpecimen]

            #magage noSuccessCount
            improvementThisGeneration = False
            for i in range(0, len(speciatedList)):
                if speciatedList[i][0].fitness > bestFitness:
                    bestFitness = speciatedList[i][0].fitness
                    bestIndividual = speciatedList[i][0]
                    improvementThisGeneration = True
            if not improvementThisGeneration:
                noSuccessCounter +=1
            else: noSuccessCounter = 0
            if noSuccessCounter == 20:
               speciatedList = cataclysm(speciatedList)
               noSuccessCounter = 0
            
            #get best fitnesvalue of this generation -> bftg
            bftg = 0
            for species in speciatedList:
                if species[0].fitness > bftg: bftg = species[0].fitness

            #printStatistics(speciatedList)
            statisticLog.update(speciatedList)

            #end the loop if either critereon is met
            if  generationCounter == 1000 or bestFitness > targetFitness: #this is the only exit from the evolutionary loop
                break

            #reproduce into new population
            population = reproduceProportional(speciatedList, self.innovationLog, 0.5, 0.66, 0.9, 0)
            for individual in population:
                individual.fitness = nn.evaluateFitness(targetNodeNR, individual, self.inputSet, self.targetSet)
            print("Evaluated generation " + str(generationCounter))
            generationCounter += 1
        print("NEAT done. Best Fitness is " + str(bestIndividual.fitness))

        plt.plot(statisticLog.generationList, statisticLog.bestFitnessList)
        plt.ylabel('Best Fitness')
        plt.xlabel('Generation')
        plt.show()

        return bestIndividual

    def basicNeat(self):
        startingIndividual = self.startingIndividual
        bestIndividual = startingIndividual

        #initialize constants
        generationCounter = 0
        noSuccessCounter = 0
        bestFitness = 0
        maximumFitness = len(self.targetSet)
        targetError = 1e-1
        targetFitness = maximumFitness - targetError

        #get outputNode TODO: adapt for multiple nodes // iteriert über die dictionary-keys (nodeNRs) und bleibt beim passenden stehen
        for targetNodeNR in startingIndividual.nodes:
            if startingIndividual.nodes[targetNodeNR].type == "output":
                break

        startingIndividual.fitness = nn.evaluateFitness(targetNodeNR, startingIndividual, self.inputSet, self.targetSet)
        
        #initialize population
        population = []
        for i in range(0,100):
            population.append(copy.deepcopy(startingIndividual))
        
        print("starting NEAT...")
        print("speciesCount; avgWeightBiasSum; avgConnectionCount; avgNodeCount; avgFitness; bestWeightBiasSum; bestConnectionCount; bestNodeCount; bestFitness; worstWeightBiasSum; worstConnectionCount; worstNodeCount; worstFitness")

        while True:
            distance = 3 
            parameters = (1,1,0.4)
            speciatedList = speciate(population, distance, parameters)
            
            population = reproduceProportional(speciatedList, self.innovationLog, 0.03, 0.05, 0.8, 0)

            noSuccessCounter+=1
            for individual in population:
                individual.writeIndividualToFile()
                individual.fitness = nn.evaluateFitness(targetNodeNR, individual, self.inputSet, self.targetSet)
                if individual.fitness > bestIndividual.fitness:
                    bestIndividual = individual
                    noSuccessCounter = 0

            if noSuccessCounter >= 20:
                speciatedList = cataclysm(speciatedList)
                noSuccessCounter = 0
            print("G=%i   P=%i   S=%i   BF=%f   noS=%i" %(generationCounter, len(population), len(speciatedList), bestIndividual.fitness, noSuccessCounter))

            if generationCounter == 500 or bestIndividual.fitness >= targetFitness:
                break
            generationCounter += 1
            


        return bestIndividual