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
        randomProblemLenght = int(random.uniform(0, maxProblemInputLenght))
        print("random lenght number: ", randomProblemLenght)

        if randomProblemLenght % 3 == 0 and randomProblemLenght != 0:
            dividableByThree = True
            print("generated problem lenght is: ", randomProblemLenght)

    numberOfSubsets = int(randomProblemLenght / 3)
    while not sumDividableByNumberOfSubsets:
        problem.clear()
        for i in range(0, randomProblemLenght):
            problem.append(int(random.uniform(0, maxProblemInputValue)))
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
    finalSum = set()
    it = iter(solution)
    for a, b, c in zip(it, it, it):
        finalSum.add(problem[a] + problem[b] + problem[c])
    return len(finalSum)


def FullSearch(goalFunction, problem, printSolutionFunction):
    potentialSolution = list(range(0, len(problem)))
    currentBestSolution = potentialSolution
    iterationIndex = 0
    for newPotentialSolution in itertools.permutations(potentialSolution):
        if (goalFunction(newPotentialSolution) < goalFunction(currentBestSolution)):
            currentBestSolution = newPotentialSolution
        printSolutionFunction(iterationIndex, currentBestSolution, goalFunction)
        iterationIndex + 1
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




def printSolution(iterationIndex, currentBestSolution, goalFunction):
    print("" + str(iterationIndex) + " " + str(goalFunction(currentBestSolution)))

generatedProblem = GenerateProblem()
#goal_function(sol1, exampleProblem)
#goal_function(generate_first_solution(generatedProblem), generatedProblem)
#FullSearch(lambda s: goal_function(s, exampleProblem), exampleProblem, printSolution)
#generate_first_solution(exampleProblem)
iterations = 10000


#hillClimbingRandomized(lambda s: GoalFunction(s, exampleProblem), lambda: GenerateFirstRandomSolution(len(exampleProblem)), getRandomNeighbour, iterations, printSolution)
hillClimbingDeterministic(lambda s: GoalFunction(s, exampleProblem), lambda: GenerateFirstRandomSolution(len(exampleProblem)), getBestNeighbour, iterations, printSolution)


