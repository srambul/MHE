import sys
import math
import itertools
import random


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

generate_problem()
