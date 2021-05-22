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
    currentBestSolution = generatedFirstRandomSolution() #generating randomely rearanged problem as first solution
    for i in range(0, iterations): #foreach iteration
        newPotentialSolution = generateRandomNeighbourFunction(currentBestSolution) #new potential solution by swaping two indexes of values
        if (goalFunction(newPotentialSolution) <= goalFunction(currentBestSolution)): #comparing if goal of new potential solution is better than first solution/new best solution
            currentBestSolution = newPotentialSolution #if is better, current best solution is the previous potential solution
        printSolutionFunction(i, currentBestSolution, goalFunction) # printing info about iteration, goal "score" and subsets
    return currentBestSolution #after comaparing all potential solutions in iterations, returns solution with best goal "score"

def GetRandomNeighbour(currentBestSolution):
    isIndexTheSame = False
    randomValueIndexToSwap = int(random.uniform(0, len(currentBestSolution) - 1)) #random index of neighbour eg [4]
    while isIndexTheSame == True:
        randomNeghbourIndex = int(random.uniform(0, len(currentBestSolution) - 1)) == randomValueIndexToSwap
        if(randomNeghbourIndex == randomValueIndexToSwap):
            isIndexTheSame=True
    bestSolutionCopy = copy.deepcopy(currentBestSolution) #copy of currentBestSolution - not reference
    bestSolutionCopy[(randomValueIndexToSwap + 1) % len(currentBestSolution)] = currentBestSolution[randomValueIndexToSwap] #eg bestSolutionCopy[(4+1%9)=5]=currentBestSolution[4]
    bestSolutionCopy[randomValueIndexToSwap] = currentBestSolution[(randomValueIndexToSwap + 1) % len(currentBestSolution)] #eg bestSolutionCopy[4]=currentBestSolution[sam as above = 5]
    return bestSolutionCopy #returns solution with swaped two indexes of values

def HillClimbingDeterministic(goalFunction, generatedFirstRandomSolution, generatedBestNeighbour, iterations, printSolutionFunction):
    currentBestSolution = generatedFirstRandomSolution() #generating randomely rearanged problem as first solution
    for i in range(0, iterations): #foreach iteration
        newPotentialSolution = generatedBestNeighbour(currentBestSolution, goalFunction) #new potential solution by selecting best neighbour
        if (newPotentialSolution == currentBestSolution): #if we approach the same potential solution as current best, print and return the currentBestSolution
            printSolutionFunction(i, currentBestSolution, goalFunction)
            return currentBestSolution
        currentBestSolution = newPotentialSolution #if is different, potential solution is now the current best solution 
        printSolutionFunction(i, currentBestSolution, goalFunction) # printing info about iteration, goal "score" and subsets
    return currentBestSolution #if all iterations didnt return newPotentialSolution wchich would be identical as currentBestSolution, just return currentBestSolution

def GetBestNeighbour(currentBestSolution, goalFunction):
    for index in range(0, len(currentBestSolution)-1): #for so many times is the lenght of problem - (for-element)
        bestSolutionCopy = copy.deepcopy(currentBestSolution) #full copy of current best solution
        bestSolutionCopy[(index + 1) % len(currentBestSolution)] = currentBestSolution[index] #bestSolutionCopy[for-element+1 % lenght of problem(eg 4)]=currentBestSolution[for-element(eg 3)]
        bestSolutionCopy[index] = currentBestSolution[(index + 1) % len(currentBestSolution)] #bestSolutionCopy[for-element(eg 3)]=currentBestSolution[for-element+1 % lenght of problem(eg 4)]
        if (goalFunction(bestSolutionCopy) <= goalFunction(currentBestSolution)): #comparing goal 'score' of bestSolutionCopy with currentBestSolution
            currentBestSolution = bestSolutionCopy #if better, bestSolutionCopy is our new currentBestSolution -> iterate with next forward neighbour
    return currentBestSolution #after comparing all values in for indexes return the best solution

def PrintSolution(iterationIndex, currentBestSolution, goalFunction):
    print("" + str(iterationIndex) + " | Score distance: " + str(goalFunction(currentBestSolution)))

def SimAnnealing(goalFunction, generatedFirstRandomSolution, generatedRandomNeighbour, temperature, iterations, printSolutionFunction):
    currentBestSolution = generatedFirstRandomSolution() #generating randomely rearanged problem as first solution
    allBestSolutions = [currentBestSolution] #new list with all currentBestSolutions during operations
    for temperatureDivider in range(1, iterations+1): #foreach iteration
        newPotentialSolution = generatedRandomNeighbour(currentBestSolution) #new potential solution by swaping two indexes of values
        if (goalFunction(newPotentialSolution) <= goalFunction(currentBestSolution)): #comparing if goal of new potential solution is better or equal than first solution/new best solution
            currentBestSolution = newPotentialSolution #if is better or equal, current best solution is the previous potential solution
            allBestSolutions.append(currentBestSolution) #add new best solution to list of all best solutions 
        else: # if newPotentialSolution was worse
            e = math.exp(- abs(goalFunction(newPotentialSolution) - goalFunction(currentBestSolution))/temperature(temperatureDivider))
            probability = random.uniform(0.0,1.0)
            if (probability < e):
                currentBestSolution = newPotentialSolution
                allBestSolutions.append(currentBestSolution)
        printSolutionFunction(temperatureDivider-1, currentBestSolution, goalFunction) # printing info about iteration, goal "score" and subsets
    return min(allBestSolutions, key=goalFunction) #return the best solution with best goal "score" in allBestSolutions list

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
        finalSolution = FullSearch(lambda s: GoalFunction(s, exampleProblem), exampleProblem, PrintSolution)
    if current_argument in ("--hillclimbingdeterministic"):
        print ("hillClimbingDeterministic chosen ")
        finalSolution = HillClimbingDeterministic(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), GetBestNeighbour, iterations, PrintSolution)
    if current_argument in ("--hillclimbingrandomized"):
        print ("hillClimbingRandomized chosen ")
        finalSolution = HillClimbingRandomized(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), GetRandomNeighbour, iterations, PrintSolution)   
    if current_argument in ("--simannealing"):
        print ("simAnnealing chosen ")
        finalSolution = SimAnnealing(lambda s: GoalFunction(s, generatedProblem), lambda: GenerateFirstRandomSolution(len(generatedProblem)), GetRandomNeighbour, lambda k : 1000.0/k, iterations, PrintSolution)

print("------------------------------------------------")
print("initial problem: " + str(generatedProblem))
print("final solution: " + str(finalSolution))
print("Final goal 'score/distance': " + str(GoalFunction(finalSolution,generatedProblem)))


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
