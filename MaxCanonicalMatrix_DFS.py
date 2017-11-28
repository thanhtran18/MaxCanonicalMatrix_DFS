# This program determines if a matrix is in its canonical form. That is, the integer associated with the given matrix is
# the largest possible integer over all its row and column permutations using DFS with pruning technique
# Author: Cong Thanh Tran.


import numpy as np
import math
import itertools
import timeit
import copy


# an instance of this class is a node of matrix, which contains necessary information to do DFS search
class NodeOfMatrix:
    def __init__(self, currMatrix, rowIndices):
        self.currMatrix = currMatrix
        self.visited = False
        self.rowIndices = rowIndices

    def set_visited(self, newVisited):
        self.visited = newVisited

    def set_row_indices(self, newRowIndices):
        self.rowIndices = newRowIndices



def isSecondLarger(matrix1, matrix2):
    exitFlag = False
    secondIsLarger = False
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if matrix2[i, j] == 1 and matrix1[i, j] == 0:
                exitFlag = True
                secondIsLarger = True
                break
            elif matrix2[i, j] == 0 and matrix1[i, j] == 1:
                exitFlag = True
                break
        if exitFlag:
            break
    return secondIsLarger


def isCanonicalDFS(givenMatrix):
    rootNode = NodeOfMatrix([], [])  # current, parent, rowIndices
    rowIndices = []
    visitedSates = set()
    states = []
    states.append(rootNode) # states of Nodes, not just the matrix
    # visitedSates.add(rootNode.currMatrix)

    result = True
    count = 1
    currNode = rootNode
    while len(states) > 0:
        temp = states.pop()  # temp is a node
        if result:
            if not temp.visited:
                temp.set_visited(True)
                currChildren = generateChildrenMatrices(givenMatrix, temp)
                # for child in currChildren:  # child is a NodeOfMatrix
                #    if not child.visited:
                #        states.append(child)
                if len(currChildren) > 0:
                    states.append(currChildren[0])
                for child in currChildren:
                    sortedChildByCols = sortColsInDescending(child.currMatrix)
                    givenTop = givenMatrix[0:len(sortedChildByCols)]
                    if np.array_equal(sortedChildByCols, givenTop):
                        for ii in range(len(currChildren)):  # child is a NodeOfMatrix
                            if ii == 0:
                                continue
                            if not currChildren[ii].visited:
                                states.append(currChildren[ii])
                        currNode = child
                        break
                    if isSecondLarger(givenTop, sortedChildByCols):
                        return False
                    else:
                        continue
    return True



# rowIndices are indices of the rows in the current
# return a list of NodeOfMatrix
def generateChildrenMatrices(givenMatrix, currentNode):
    children = []
    indexList = copy.deepcopy(currentNode.rowIndices)
    currHolder = copy.deepcopy(currentNode.currMatrix)
    for i in range(len(givenMatrix)):
        if i not in indexList:
            indexList.append(i)
            if len(currentNode.currMatrix) > 0:
                newMatrix = np.append(currHolder, [givenMatrix[i]], axis=0)

            else:
                currHolder.append(givenMatrix[i])
                newMatrix = np.array(currHolder)

            children.append(NodeOfMatrix(np.array(newMatrix), indexList))
            currHolder = copy.deepcopy(currentNode.currMatrix)
            indexList = copy.deepcopy(currentNode.rowIndices)
    return children


# sort rows of the given matrix in descending order
def sortRowsInDescending(givenMatrix):
    # sort in ascending order
    givenMatrix = np.apply_along_axis(lambda r: ''.join([str(c) for c in r]), 1, givenMatrix)
    givenMatrix.sort(axis=0)
    givenMatrix = np.array([[int(i) for i in r] for r in givenMatrix])
    # get the sorted version in descending order
    givenMatrix = givenMatrix[::-1]
    return givenMatrix

# sort columns of the given matrix in descending order
def sortColsInDescending(givenMatrix):
    # sort in ascending order
    givenMatrix = np.apply_along_axis(lambda r: ''.join([str(c) for c in r]), 0, givenMatrix)
    givenMatrix.sort()
    givenMatrix = np.array([[int(i) for i in r] for r in givenMatrix])
    # get the descending version
    givenMatrix = givenMatrix.T
    givenMatrix = np.fliplr(givenMatrix)
    return givenMatrix


def generateRowPermutations(givenMatrix):
    permutations = []
    duplicate = False
    numOfPermutations = 0
    while numOfPermutations < math.factorial(len(givenMatrix)):
        newMatrix = np.random.permutation(givenMatrix)
        for curr in permutations:
            if np.array_equal(curr, newMatrix):
                duplicate = True
            else:
                duplicate = False

        if not duplicate:
            permutations.append(newMatrix)
            numOfPermutations += 1
    return permutations


# distinct: number of distinct numbers (from 1 to distinct)
def convertInput(inputMatrix, distinct, numRows):
    convertedMatrix = [[0 for x in range(distinct)] for y in range(numRows)]
    for i in range(len(inputMatrix)):
        for j in range(len(inputMatrix[0])):
            convertedMatrix[i][inputMatrix[i][j] - 1] = 1
    return np.matrix(convertedMatrix)


# TESTING MATRICES
# matrix = np.matrix([[1,1,1,0,0],[1,1,0,1,0],[0,0,1,1,1]])

# matrix = np.array([[1,1,1,0,0],[1,0,0,1,0],[0,1,0,0,1]])
# matrix = np.array([[1,1,1,0,0,1,1,0,1],[1,1,1,0,0,1,0,0,1],[1,1,0,1,0,0,1,1,0],[1,1,0,1,0,0,1,0,0],[1,0,1,1,0,0,1,1,0],[1,0,0,1,0,0,1,0,0]]) #false
# matrix = np.array([[1,1,1,1,1,1,0,0,0],[1,1,1,1,1,0,0,0,0],[1,1,0,0,0,1,1,1,0],[1,1,0,0,0,1,1,0,0],[1,0,1,0,0,1,1,1,0],[1,0,0,0,0,1,1,0,0]]) #true
# matrix = np.array([[1,1,1,1,1,1,0,0,0],[1,1,1,1,1,0,0,0,0],[1,1,0,0,0,1,1,1,0],[1,1,0,0,0,1,1,0,0],[1,0,1,0,0,1,1,1,0],[1,0,0,0,0,1,1,0,0],[1,0,0,0,0,1,0,1,1],[1,0,0,0,0,1,0,1,0],[1,0,0,0,0,0,0,0,0]])

# 9 rows, same num of 1s, false
# matrix = np.array([[1,1,1,1,1,0,0,0,0],[1,1,1,1,0,1,0,0,0],[1,1,0,0,0,1,1,1,0],[1,1,0,0,0,1,1,0,1],[1,0,1,0,0,1,1,1,0],[1,0,0,0,1,1,1,1,0],[1,0,0,0,1,1,1,0,1],[1,0,0,0,0,1,1,1,1],[1,0,0,0,0,1,1,1,1]])
# 9 rows, same num of 1s, false
# matrix = np.array([[1,0,0,0,0,1,1,1,1],[1,0,0,0,0,1,1,1,1],[1,1,1,1,1,0,0,0,0],[1,1,1,1,0,1,0,0,0],[1,1,0,0,0,1,1,1,0],[1,1,0,0,0,1,1,0,1],[1,0,1,0,0,1,1,1,0],[1,0,0,0,1,1,1,1,0],[1,0,0,0,1,1,1,0,1]])
# 8 rows, same num of 1s, true
# matrix = np.array([[1,1,1,1,1,0,0,0,0],[1,1,1,1,0,1,0,0,0],[1,1,1,1,0,0,1,0,0],[1,1,1,1,0,0,0,1,0],[1,1,1,0,1,1,0,0,0],[1,1,0,0,1,0,1,1,0],[1,0,1,0,1,0,1,0,1],[1,0,0,0,1,0,1,1,1]])
# 7 rows, same num of 1s, true
# matrix = np.array([[1,1,1,1,1,0,0,0,0],[1,1,1,1,0,1,0,0,0],[1,1,1,1,0,0,1,0,0],[1,1,1,1,0,0,0,1,0],[1,1,1,0,1,1,0,0,0],[1,1,0,0,1,0,1,1,0],[1,0,1,0,1,0,1,0,1]])

# matrix = np.array([[1,1,1,1,0,0,0,1,0],[1,1,1,0,1,1,0,0,0],[1,1,0,0,1,0,1,1,0],[1,0,1,0,1,0,1,0,1]])
# matrix = np.array([[1,1,1,0,0,0,0],[1,1,0,1,0,0,0],[1,0,1,0,1,0,0],[1,0,1,0,0,1,0],[1,0,0,1,0,0,1],[1,0,0,0,1,1,0],[0,0,1,1,1,0,0]])
# matrix = np.array([[1,1,1,1,1,1,0,0,0],[1,1,1,1,1,0,0,0,0],[1,1,0,0,0,1,1,1,0],[1,1,0,0,0,1,1,0,0],[1,0,1,0,0,1,1,1,0],[1,0,0,0,0,1,1,0,0],[1,0,0,0,0,1,0,1,1]])

# matrix = np.array([[1,1,1,0,0],[1,0,0,1,1],[0,1,1,0,1]])  # should be false
# matrix = np.array([[1,1,1,0,0],[1,0,0,1,1],[0,1,0,1,1]])  # should be false
# MAIN PROGRAM STARTS HERE
# matrix = np.array([[1,1,0,1,0,0,0],[0,1,1,0,1,0,0],[0,0,1,1,0,1,0],[0,0,0,1,1,0,1]])

print("Please enter your matrix, with spaces between the columns.")
print("Use one row per line, and a blank row when you're done.")
matrix = []
numLine = 0
initial = []
while True:
    numLine += 1
    line = input()
    if numLine == 1:
        initial = line.split()
    else:
        if not line:
            break
        values = line.split()
        row = [int(value) for value in values]
        matrix.append(row)

cols = int(initial[0])
rows = int(initial[1])
matrix = convertInput(matrix, cols, rows)

start = timeit.default_timer()


inputMatrix = np.array(matrix)


print(inputMatrix)
if isCanonicalDFS(inputMatrix):
    print("True. The given matrix is in its canonical form.")
else:
    print("False. The given matrix is NOT in its canonical form.")

stop = timeit.default_timer()
print(stop - start, "s")
