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

def generate_problem():
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

def goal_function(solution, problem):
    finalSum = set()
    it = iter(solution)
    for a, b, c in zip(it, it, it):
        finalSum.add(problem[a] + problem[b] + problem[c])
    return len(finalSum)


def fullSearch(goal, problem, onIteration):
    f = list(range(0, len(problem)))
    currentBest = f
    i = 0
    for newSol in itertools.permutations(f):
        if (goal(newSol) < goal(currentBest)):
            currentBest = newSol
        onIteration(i, currentBest, goal)
        i = i + 1
    return currentBest

def printSolution(i, currentBest, goal):
    print("" + str(i) + " " + str(goal(currentBest)))


def hillClimbingRandomized(goal, gensol, genNeighbour, iterations, onIteration):
    '''goal - funkcja celu (to optymalizujemy),
    gensol - generowanie losowego rozwiazania,
    genNeighbour - generowanie losowego punktu z otoczenia rozwiazania,
    iterations - liczba iteracji alg.'''
    currentBest = gensol()
    for i in range(0, iterations):
        newSol = genNeighbour(currentBest)
        if (goal(newSol) <= goal(currentBest)):
            currentBest = newSol
        onIteration(i, currentBest, goal)
    return currentBest

def hillClimbingDeterministic(goal, gensol, genBestNeighbour, iterations,onIteration):
    '''goal - funkcja celu (to optymalizujemy),
    gensol - generowanie losowego rozwiazania,
    genNeighbour - generowanie losowego punktu z otoczenia rozwiazania,
    iterations - liczba iteracji alg.'''
    currentBest = gensol()
    for i in range(0, iterations):
        newSol = genBestNeighbour(currentBest, goal)
        if (newSol == currentBest):
            return currentBest
        currentBest = newSol
        onIteration(i, currentBest, goal)
    return currentBest

def getBestNeighbour(currPoint, goal):
    best = currPoint
    for swapPoint in range(0,len(currPoint)-1):
        newPoint = copy.deepcopy(currPoint)
        newPoint[(swapPoint+1) % len(currPoint)] = currPoint[swapPoint]
        newPoint[swapPoint] = currPoint[(swapPoint+1) % len(currPoint)]
        if (goal(newPoint) <= goal(best)):
            best = newPoint
    return best

def getRandomNeighbour(currPoint):
    swapPoint = int(random.uniform(0, len(currPoint) - 1))
    newPoint = copy.deepcopy(currPoint)
    newPoint[(swapPoint + 1) % len(currPoint)] = currPoint[swapPoint]
    newPoint[swapPoint] = currPoint[(swapPoint + 1) % len(currPoint)]
    return newPoint

def generateRandomSolution(n):
    r = []
    f = list(range(0, n))
    for i in range(0, n):
        p = int(random.uniform(0, len(f) - 1))
        r.append(f[p])
        del f[p]
        # print(f)
        # print(r)
    return r


generatedProblem = generate_problem()
#goal_function(sol1, exampleProblem)
#goal_function(generate_first_solution(generatedProblem), generatedProblem)
#fullSearch(lambda s: goal_function(s, exampleProblem), exampleProblem, printSolution)
#generate_first_solution(exampleProblem)
iterations = 1000000


hillClimbingRandomized(lambda s: goal_function(s, generatedProblem), lambda: generateRandomSolution(len(generatedProblem)), getRandomNeighbour, iterations, printSolution)
#hillClimbingDeterministic(lambda s: goal_function(s, generatedProblem), lambda: generateRandomSolution(len(generatedProblem)), getBestNeighbour, iterations, printSolution)


