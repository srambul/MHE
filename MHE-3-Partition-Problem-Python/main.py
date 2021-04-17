import sys
import math
import itertools
import random

exampleProblem = [1, 2, 3, 4, 5, 6, 7, 8, 9]
exampleSolution = [[9, 3, 2], [8, 5, 1], [7, 3, 4]]

if len(sys.argv) > 1:
    maxProblemInputValue = sys.argv[1]
else:
    maxProblemInputValue = 20

if len(sys.argv) > 1:
    maxProblemInputLenght = sys.argv[2]
else:
    maxProblemInputLenght = 20

def generate_problem():
    dividableByThree = False
    problem = []
    while not dividableByThree:
        randomProblemLenght = int(random.uniform(0, maxProblemInputLenght))
        print("random lenght number: ", randomProblemLenght)

        if randomProblemLenght % 3 == 0 and randomProblemLenght != 0:
            dividableByThree = True
            print("generated problem lenght is: ", randomProblemLenght)

    for i in range(0, randomProblemLenght):
        problem.append(int(random.uniform(0, maxProblemInputValue)))
    print("Our problem: ", problem)
    return problem

def goal_function(solution, problem):
    finalSum = set()
    for subSet in solution:
        finalSum.add(sum(subSet))
    if len(finalSum) == 1:
        print("solution: ", solution, " is correct for problem: ", problem)
    else:
        print("solution: ", solution, " is not correct for problem: ", problem, "with sums of subsets: ", finalSum)

def generate_first_solution(problem):
    numberOfSubsets = int(len(problem)/3)
    solution = []
    for i in range(0, numberOfSubsets):
        solution.append([problem[i], problem[i+1], problem[i+2]])

    print(solution)
    return solution

generatedProblem = generate_problem()
#goal_function(exampleSolution, exampleProblem)
goal_function(generate_first_solution(generatedProblem), generatedProblem)
