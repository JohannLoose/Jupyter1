import NeuralNets as nn

def setPredecessorConecctions(individual):
    for conNr in individual.connections:
        individual.nodes[individual.connections[conNr].end].addPredecessorConnection(individual.connections[conNr])

class XorSet:
    def __init__(self):
        self.inputSet = [(0,0),
                         (0,1),
                         (1,0),
                         (1,1)]
        self.targetSet = [0, 1, 1, 0]

        self.startingIndividual = nn.Individual()
        self.startingIndividual.addNode(nn.Node(0,"input"))
        self.startingIndividual.addNode(nn.Node(1,"input"))
        self.startingIndividual.addNode(nn.Node(2,"output"))
        self.startingIndividual.addConnection(nn.Connection(0,2,0,0))
        self.startingIndividual.addConnection(nn.Connection(1,2,0,1))

        setPredecessorConecctions(self.startingIndividual)

    def returnCanonicalIndividual(self):
        individual = nn.Individual()
        individual.addNode(nn.Node(0,"input"))
        individual.addNode(nn.Node(1,"input"))
        individual.addNode(nn.Node(2,"output"))
        individual.addNode(nn.Node(3,"hidden"))
        individual.addConnection(nn.Connection(0,3,0,0))
        individual.addConnection(nn.Connection(1,3,0,1))
        individual.addConnection(nn.Connection(0,2,0,2))
        individual.addConnection(nn.Connection(1,2,0,3))
        individual.addConnection(nn.Connection(3,2,0,4))
        setPredecessorConecctions(individual)
        return individual

    def altIndividual(self):
        individual = nn.Individual()
        individual.addNode(nn.Node(0,"input"))
        individual.addNode(nn.Node(1,"input"))
        individual.addNode(nn.Node(2,"output"))
        individual.addNode(nn.Node(3,"hidden"))
        individual.addNode(nn.Node(4,"hidden"))
        individual.addConnection(nn.Connection(0,3,0,0))
        individual.addConnection(nn.Connection(1,3,0,1))
        individual.addConnection(nn.Connection(0,4,0,2))
        individual.addConnection(nn.Connection(1,4,0,3))
        individual.addConnection(nn.Connection(3,2,0,4))
        individual.addConnection(nn.Connection(4,2,0,5))
        setPredecessorConecctions(individual)
        return individual

class ThreeBitParitySet:
    def __init__(self):
        self.inputSet = [(0,0,0),
                         (0,0,1),
                         (0,1,0),
                         (0,1,1),
                         (1,0,0),
                         (1,0,1),
                         (1,1,0),
                         (1,1,1)]
        self.targetSet = [0, 1, 1, 0, 1, 0, 0, 1]

        self.startingIndividual = nn.Individual()
        self.startingIndividual.addNode(nn.Node(0,"input"))
        self.startingIndividual.addNode(nn.Node(1,"input"))
        self.startingIndividual.addNode(nn.Node(2,"input"))
        self.startingIndividual.addNode(nn.Node(3,"output"))
        self.startingIndividual.addConnection(nn.Connection(0,3,0,0))
        self.startingIndividual.addConnection(nn.Connection(1,3,0,1))
        self.startingIndividual.addConnection(nn.Connection(2,3,0,2))

        setPredecessorConecctions(self.startingIndividual)

    def returnCanonicalIndividual(self):
        individual = nn.Individual()
        individual.addNode(nn.Node(0,"input"))
        individual.addNode(nn.Node(1,"input"))
        individual.addNode(nn.Node(2,"input"))
        individual.addNode(nn.Node(3,"output"))
        individual.addNode(nn.Node(4,"hidden"))
        individual.addNode(nn.Node(5,"hidden"))
        individual.addNode(nn.Node(6,"hidden"))
        individual.addConnection

    def returnCanonicalIndividual(self):
        individual = nn.Individual()
        individual.addNode(nn.Node(0,"input"))
        individual.addNode(nn.Node(1,"input"))
        individual.addNode(nn.Node(2,"input"))
        individual.addNode(nn.Node(3,"output"))
        individual.addNode(nn.Node(4,"hidden"))
        individual.addNode(nn.Node(5,"hidden"))
        individual.addNode(nn.Node(6,"hidden"))
        individual.addConnection(nn.Connection(0,4,0,0))
        individual.addConnection(nn.Connection(1,4,0,1))
        individual.addConnection(nn.Connection(0,5,0,2))
        individual.addConnection(nn.Connection(1,5,0,3))
        individual.addConnection(nn.Connection(4,5,0,4))
        individual.addConnection(nn.Connection(5,6,0,5))
        individual.addConnection(nn.Connection(2,6,0,6))
        individual.addConnection(nn.Connection(5,3,0,7))
        individual.addConnection(nn.Connection(6,3,0,8))
        individual.addConnection(nn.Connection(2,3,0,9))
        setPredecessorConecctions(individual)
        return individual
    
    def altIndividual(self):
        individual = nn.Individual()


    

class FourBitParitySet:
    def __init__(self):
        self.inputSet = [(0,0,0,0),
                         (0,0,0,1),
                         (0,0,1,0),
                         (0,0,1,1),
                         (0,1,0,0),
                         (0,1,0,1),
                         (0,1,1,0),
                         (0,1,1,1),
                         (1,0,0,0),
                         (1,0,0,1),
                         (1,0,1,0),
                         (1,0,1,1),
                         (1,1,0,0),
                         (1,1,0,1),
                         (1,1,1,0),
                         (1,1,1,1)]
        self.targetSet = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]

        self.startingIndividual = nn.Individual()
        self.startingIndividual.addNode(nn.Node(0,"input"))
        self.startingIndividual.addNode(nn.Node(1,"input"))
        self.startingIndividual.addNode(nn.Node(2,"input"))
        self.startingIndividual.addNode(nn.Node(3,"input"))
        self.startingIndividual.addNode(nn.Node(4,"output"))
        self.startingIndividual.addConnection(nn.Connection(0,4,0,0))
        self.startingIndividual.addConnection(nn.Connection(1,4,0,1))
        self.startingIndividual.addConnection(nn.Connection(2,4,0,2))
        self.startingIndividual.addConnection(nn.Connection(3,4,0,3))

        setPredecessorConecctions(self.startingIndividual)
