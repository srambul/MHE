import copy
import getopt, sys
import math
import itertools
import random
import json

def GenerateProblem():
    dividableByThree = False
    sumDividableByNumberOfSubsets = False
    problem = []
    while not dividableByThree:
        randomProblemLenght = int(random.uniform(9, maxProblemInputLenght))
        print("random lenght number: ", randomProblemLenght)
        if randomProblemLenght % 3 == 0 and randomProblemLenght != 0:
            dividableByThree = True
            print("generated problem lenght is: ", randomProblemLenght)

    numberOfSubsets = int(randomProblemLenght / 3)
    while not sumDividableByNumberOfSubsets:
        problem.clear()
        for i in range(0, randomProblemLenght):
            problem.append(int(random.uniform(1, maxProblemInputValue)))
        print("Proposed problem: ", problem)
        if sum(problem) % numberOfSubsets == 0:
            sumDividableByNumberOfSubsets = True
        else:
            print("Proposed above problem does not qualify for calculations")

    print("Our problem: ", problem)
    return problem

def GenerateFirstRandomSolution(problemLenght):
    generatedFirstRandomSolution = []
    problemIndexList = list(range(0, problemLenght))
    for i in range(0, problemLenght):
        randomProblemIndex = int(random.uniform(0, len(problemIndexList) - 1))
        generatedFirstRandomSolution.append(problemIndexList[randomProblemIndex])
        del problemIndexList[randomProblemIndex]
    return generatedFirstRandomSolution

def GoalFunction(solution, problem):
    numberOfSubsets = int(len(problem) / 3)
    it = iter(solution)
    perfectSum = int(sum(problem) / numberOfSubsets)
    print("perfect sum: " + str(perfectSum))
    subSets = set()
    for a, b, c in zip(it, it, it):
        subSets.add(problem[a] + problem[b] + problem[c])
        print("subset:" + str(problem[a]) + "," + str(problem[b]) + "," + str(problem[c]) + " Sum: " + str(problem[a] + problem[b] + problem[c]))
    score = 0
    for subSet in subSets:
        score = score + abs(perfectSum - subSet)
    print("Distance: " + str(score))
    return score
    
def FullSearch(goalFunction, problem, printSolutionFunction):
    potentialSolution = list(range(0, len(problem)))
    currentBestSolution = potentialSolution
    iterationIndex = 0
    for newPotentialSolution in itertools.permutations(potentialSolution):
        if (goalFunction(newPotentialSolution) < goalFunction(currentBestSolution)):
            currentBestSolution = newPotentialSolution
        printSolutionFunction(iterationIndex, currentBestSolution, goalFunction)
        iterationIndex = iterationIndex + 1
    return currentBestSolution

def HillClimbingRandomized(goalFunction, generatedFirstRandomSolution, generateRandomNeighbourFunction, iterations, printSolutionFunction):
    currentBestSolution = generatedFirstRandomSolution()
    for i in range(0, iterations):
        newPotentialSolution = generateRandomNeighbourFunction(currentBestSolution)
        if (goalFunction(newPotentialSolution) <= goalFunction(currentBestSolution)):
            currentBestSolution = newPotentialSolution
        printSolutionFunction(i, currentBestSolution, goalFunction)
    return currentBestSolution

def HillClimbingDeterministic(goalFunction, generatedFirstRandomSolution, generatedBestNeighbour, iterations, printSolutionFunction):
    currentBestSolution = generatedFirstRandomSolution()
    for i in range(0, iterations):
        newPotentialSolution = generatedBestNeighbour(currentBestSolution, goalFunction)
        if (newPotentialSolution == currentBestSolution):
            return currentBestSolution
        currentBestSolution = newPotentialSolution
        printSolutionFunction(i, currentBestSolution, goalFunction)
    return currentBestSolution

def GetBestNeighbour(currentBestSolution, goalFunction):
    currentBestSolution
    for index in range(0, len(currentBestSolution)-1):
        bestSolutionCopy = copy.deepcopy(currentBestSolution)
        bestSolutionCopy[(index + 1) % len(currentBestSolution)] = currentBestSolution[index]
        bestSolutionCopy[index] = currentBestSolution[(index + 1) % len(currentBestSolution)]
        if (goalFunction(bestSolutionCopy) <= goalFunction(currentBestSolution)):
            currentBestSolution = bestSolutionCopy
    return currentBestSolution

def GetRandomNeighbour(currentBestSolution):
    randomSolutionIndex = int(random.uniform(0, len(currentBestSolution) - 1))
    bestSolutionCopy = copy.deepcopy(currentBestSolution)
    bestSolutionCopy[(randomSolutionIndex + 1) % len(currentBestSolution)] = currentBestSolution[randomSolutionIndex]
    bestSolutionCopy[randomSolutionIndex] = currentBestSolution[(randomSolutionIndex + 1) % len(currentBestSolution)]
    return bestSolutionCopy

def GetRandomNeighbour2(currentBestSolution):
    for i in range(0, int(min(abs( random.normalvariate(0.0,2.0)) + 1,500)) ):
        randomSolutionIndex = int(random.uniform(0, len(currentBestSolution)-1))
        bestSolutionCopy = copy.deepcopy(currentBestSolution)
        bestSolutionCopy[(randomSolutionIndex + 1) % len(currentBestSolution)] = currentBestSolution[randomSolutionIndex]
        bestSolutionCopy[randomSolutionIndex] = currentBestSolution[(randomSolutionIndex + 1) % len(currentBestSolution)]
        currentBestSolution = bestSolutionCopy
    return currentBestSolution

def PrintSolution(iterationIndex, currentBestSolution, goalFunction):
    print("" + str(iterationIndex) + " | Score distance: " + str(goalFunction(currentBestSolution)))

def SimAnnealing(goalFunction, generatedFirstRandomSolution, generatedBestNeighbour, temperature, iterations, printSolutionFunction):
    currentBestSolution = generatedFirstRandomSolution()
    allBestSolutions = [currentBestSolution]
    for i in range(1, iterations+1):
        newPotentialSolution = generatedBestNeighbour(currentBestSolution)
        if (goalFunction(newPotentialSolution) <= goalFunction(currentBestSolution)):
            currentBestSolution = newPotentialSolution
            allBestSolutions.append(currentBestSolution)
        else:
            e = math.exp(- abs(goalFunction(newPotentialSolution) - goalFunction(currentBestSolution))/temperature(i))
            u = random.uniform(0.0,1.0)
            if (u < e):
                currentBestSolution = newPotentialSolution
                allBestSolutions.append(currentBestSolution)
        printSolutionFunction(i-1, currentBestSolution, goalFunction)
    return min(allBestSolutions, key=goalFunction)

exampleProblem = [1, 2, 3, 4, 5, 6, 7, 8, 9]
exampleProblem2 = [1, 2, 5, 6, 7, 9]

minProblemInputValue = 1
maxProblemInputValue = 20
minProblemInputLenght = 9
maxProblemInputLenght = 20
iterations = 500
generatedProblem = exampleProblem

full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]

short_options = ""
long_options = ["minvalue=", "maxvalue=", "minlenght=", "maxlenght=", "iterations=", "generateproblem", "customproblem", "fullsearch", "hillclimbingdeterministic", "hillclimbingrandomized", "simannealing"]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    print (str(err))
    sys.exit(2)

#Input parameter options settings
for current_argument, current_value in arguments:
    if current_argument in ("--minvalue"):
        print ("Min problem element value set to " + current_value)
        minProblemInputValue = int(current_value)
    if current_argument in ("--maxvalue"):
        print ("Max problem element value set to " + current_value)
        maxProblemInputValue = int(current_value)
    if current_argument in ("--minlenght"):
        print ("Max problem lenght set to " + current_value)
        minProblemInputLenght = int(current_value)
    if current_argument in ("--maxlenght"):
        print ("Max problem lenght set to " + current_value)
        maxProblemInputLenght = int(current_value)
    if current_argument in ("--iterations"):
        print ("problem iterations were set to " + current_value)
        iterations = int(current_value)
    if current_argument in ("--generateproblem"):
        print ("generating problem ")
        generatedProblem = GenerateProblem()
    if current_argument in ("--customproblem"):
        print("getting custom problem from custom_problem.json file")
        with open("custom_problem.json") as jsonfile:
            jsonparsed = json.load(jsonfile)
        problem = jsonparsed["dataset"]
    if current_argument in ("--fullsearch"):
        print ("fullsearch chosen ")
        FullSearch(lambda s: GoalFunction(s, exampleProblem), exampleProblem, PrintSolution)
    if current_argument in ("--hillclimbingdeterministic"):
        print ("hillClimbingDeterministic chosen ")
        HillClimbingDeterministic(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), GetBestNeighbour, iterations, PrintSolution)
    if current_argument in ("--hillclimbingrandomized"):
        print ("hillClimbingRandomized chosen ")
        HillClimbingRandomized(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), GetRandomNeighbour, iterations, PrintSolution)
    if current_argument in ("--simannealing"):
        print ("simAnnealing chosen ")
        SimAnnealing(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), GetRandomNeighbour, lambda k : 1000.0/k, iterations, PrintSolution)




# for arg in sys.argv:
#     if sys.argv[1] != 0:
#         maxProblemInputValue = int(sys.argv[1])
#     if sys.argv[2] != 0:
#         maxProblemInputLenght = int(sys.argv[2])
#     else:
#         maxProblemInputLenght = 20
#     if arg == '-iterations':  
#     if arg == '-generateproblem':
#         generatedProblem = GenerateProblem()
#     if arg == '-fullsearch':
#         FullSearch(lambda s: GoalFunction(s, exampleProblem), exampleProblem, PrintSolution)
#     if arg == '-hillclimbingdeterministic':
#         hillClimbingDeterministic(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), getBestNeighbour, iterations, PrintSolution)
#     if arg == '-hillclimbingrandomized':
#         hillClimbingRandomized(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), getRandomNeighbour, iterations, PrintSolution)
#     if arg == '-simannealing':
#         simAnnealing(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), getRandomNeighbour, lambda k : 1000.0/k, iterations, PrintSolution)
