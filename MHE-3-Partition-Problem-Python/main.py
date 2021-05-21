import copy
import sys
import math
import itertools
import random
import json

exampleProblem = [1, 2, 3, 4, 5, 6, 7, 8, 9]
exampleProblem2 = [1, 2, 5, 6, 7, 9]
exampleSolution = [[9, 3, 2], [8, 5, 1], [7, 3, 4]]
sol1 = [8, 4, 0, 6, 5, 1, 2, 3, 7]
sol = [9, 5, 1, 7, 6, 2, 3, 4, 8]

if len(sys.argv) > 1:
    maxProblemInputValue = int(sys.argv[1])
else:
    maxProblemInputValue = 20

if len(sys.argv) > 2:
    maxProblemInputLenght = int(sys.argv[2])
else:
    maxProblemInputLenght = 20

def GenerateProblem():
    dividableByThree = False
    sumDividableByNumberOfSubsets = False
    problem = []
    while not dividableByThree:
        randomProblemLenght = int(random.uniform(6, maxProblemInputLenght))
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

def hillClimbingRandomized(goalFunction, generatedFirstRandomSolution, generateRandomNeighbourFunction, iterations, printSolutionFunction):
    currentBestSolution = generatedFirstRandomSolution()
    for i in range(0, iterations):
        newPotentialSolution = generateRandomNeighbourFunction(currentBestSolution)
        if (goalFunction(newPotentialSolution) <= goalFunction(currentBestSolution)):
            currentBestSolution = newPotentialSolution
        printSolutionFunction(i, currentBestSolution, goalFunction)
    return currentBestSolution

def hillClimbingDeterministic(goalFunction, generatedFirstRandomSolution, generatedBestNeighbour, iterations, printSolutionFunction):
    currentBestSolution = generatedFirstRandomSolution()
    for i in range(0, iterations):
        newPotentialSolution = generatedBestNeighbour(currentBestSolution, goalFunction)
        if (newPotentialSolution == currentBestSolution):
            return currentBestSolution
        currentBestSolution = newPotentialSolution
        printSolutionFunction(i, currentBestSolution, goalFunction)
    return currentBestSolution

def getBestNeighbour(currentBestSolution, goalFunction):
    currentBestSolution
    for index in range(0, len(currentBestSolution)-1):
        bestSolutionCopy = copy.deepcopy(currentBestSolution)
        bestSolutionCopy[(index + 1) % len(currentBestSolution)] = currentBestSolution[index]
        bestSolutionCopy[index] = currentBestSolution[(index + 1) % len(currentBestSolution)]
        if (goalFunction(bestSolutionCopy) <= goalFunction(currentBestSolution)):
            currentBestSolution = bestSolutionCopy
    return currentBestSolution

def getRandomNeighbour(currentBestSolution):
    randomSolutionIndex = int(random.uniform(0, len(currentBestSolution) - 1))
    bestSolutionCopy = copy.deepcopy(currentBestSolution)
    bestSolutionCopy[(randomSolutionIndex + 1) % len(currentBestSolution)] = currentBestSolution[randomSolutionIndex]
    bestSolutionCopy[randomSolutionIndex] = currentBestSolution[(randomSolutionIndex + 1) % len(currentBestSolution)]
    return bestSolutionCopy

def getRandomNeighbour2(currentBestSolution):
    for i in range(0, int(min(abs( random.normalvariate(0.0,2.0)) + 1,500)) ):
        randomSolutionIndex = int(random.uniform(0, len(currentBestSolution)-1))
        bestSolutionCopy = copy.deepcopy(currentBestSolution)
        bestSolutionCopy[(randomSolutionIndex + 1) % len(currentBestSolution)] = currentBestSolution[randomSolutionIndex]
        bestSolutionCopy[randomSolutionIndex] = currentBestSolution[(randomSolutionIndex + 1) % len(currentBestSolution)]
        currentBestSolution = bestSolutionCopy
    return currentBestSolution

def PrintSolution(iterationIndex, currentBestSolution, goalFunction):
    print("" + str(iterationIndex) + " " + str(goalFunction(currentBestSolution)))

def simAnnealing(goalFunction, generatedFirstRandomSolution, generatedBestNeighbour, T, iterations, printSolutionFunction):
    currentBest = generatedFirstRandomSolution()
    V = [currentBest]
    for i in range(1, iterations+1):
        newPotentialSolution = generatedBestNeighbour(currentBest)
        if (goalFunction(newPotentialSolution) <= goalFunction(currentBest)):
            currentBest = newPotentialSolution
            V.append(currentBest)
        else:
            e = math.exp(- abs(goalFunction(newPotentialSolution) - goalFunction(currentBest))/T(i))
            u = random.uniform(0.0,1.0)
            if (u < e):
                currentBest = newPotentialSolution
                V.append(currentBest)
        printSolutionFunction(i-1, currentBest, goalFunction)
    
    return min(V, key=goalFunction)

generatedProblem = GenerateProblem()
#goal_function(sol1, exampleProblem)
#goal_function(generate_first_solution(generatedProblem), generatedProblem)
FullSearch(lambda s: GoalFunction(s, exampleProblem), exampleProblem, PrintSolution)
#generate_first_solution(exampleProblem)
iterations = 1000
#hillClimbingRandomized(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), getRandomNeighbour, iterations, PrintSolution)
#hillClimbingDeterministic(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), getBestNeighbour, iterations, PrintSolution)
#finalSolution = simAnnealing(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), getRandomNeighbour, lambda k : 1000.0/k, iterations, PrintSolution)
#print(GoalFunction(finalSolution,generatedProblem))

