import numpy as np
from tqdm import tqdm
from sympy.combinatorics.graycode import GrayCode

# make the graph (hierarchy)
class Graph(object):
    def __init__(self, numNodes):
        self.adjacencyMatrix = []
        for i in range(numNodes): 
            self.adjacencyMatrix.append([0 for i in range(numNodes)])
        self.numNodes = numNodes

    def addEdge(self, start, end):
        self.adjacencyMatrix[start][end] = 1

    def removeEdge(self, start, end):
        if self.adjacencyMatrix[start][end] == 0:
            print("There is no edge between %d and %d" % (start, end))
        else:
            self.adjacencyMatrix[start][end] = 0

    def containsEdge(self, start, end):
        if self.adjacencyMatrix[start][end] > 0:
            return True
        else:
            return False

    def __len__(self):
        return self.numNodes

# 2^n concept state matrix generation
def make_all_concept_state_mat(num_of_concepts):
    concept_state = list()
    all_concept_state = sorted(list(GrayCode(num_of_concepts).generate_gray()))
    
    for state in all_concept_state:
        for i in state:
            concept_state.append(int(i))
            
    concept_state = np.reshape(concept_state,(-1,num_of_concepts))
    
    return concept_state

def NOT_GATE(input_vectors):
    result = list()
    for i in input_vectors.tolist():
        for j in range(len(i)):
            if i[j] == 0 :
                i[j] = 1
            else :
                i[j] = 0      
        result.append(i)
    return result

# ideal response (Barnes, 2005)
def make_ideal_response(concept_state, Q_matrix):
    tmp = np.dot(NOT_GATE(concept_state), Q_matrix)
    for i in tmp:
        for j in range(len(i)):
            if i[j] == 0 :
                pass
            else :
                i[j] = 1
                
    ideal_response = NOT_GATE(tmp)
    return ideal_response


# calculate hamming distance between Q matrix and Qr matrix
def calc_hamming_dist(Qmat, reduced_Qmat):
    
    hamming_distance = list()

    for i in range(Qmat.shape[0]):
        count = 0
        for j in range(reduced_Qmat.shape[0]):
            if reduced_Qmat[j] == Qmat[i][j]:
                pass
            else:
                count += 1
        hamming_distance.append(count)
    
    return hamming_distance


def calc_min_hamming_dist_between_IDR_and_response(response, IDR):
    h_dist = list()
    
    for i in range(len(IDR)):
        count = 0
        for j in range(response.shape[0]):
            if response[j] == IDR[i][j]:
                pass
            else:
                count += 1
        h_dist.append(count)
        
    return min(h_dist)

def get_index(Qr, hamming_dists, iters):
    mapping_index = np.zeros((Qr.shape[0], iters))

    for i in range(len(hamming_dists)):
        index_with_min_dist = list(np.where(np.array(hamming_dists[i]) == np.min(np.array(hamming_dists[i])))[0])

        if len(index_with_min_dist) != 1 :
            for j in range(iters):
                mapping_index[i][j] = np.random.choice(index_with_min_dist)

        else:
            mapping_index[i] = index_with_min_dist[0]
    
    return mapping_index

def calc_cost(response, IDR, Qr, indices, iters):
    mapping_response_data = np.zeros((response.shape[0], Qr.shape[0]))
    result = list()
    
    for i in tqdm(range(iters)):
        for j in range(Qr.shape[0]):
            mapping_response_data[:, j] = response[:, int(indices[:, i][j])]
            
        total = 0
        for k in range(response.shape[0]):
            total += calc_min_hamming_dist_between_IDR_and_response(mapping_response_data[k], IDR)
            
        result.append(total / (response.shape[0] * Qr.shape[0]))
    
    return result