import numpy as np
from tqdm import tqdm
from sympy.combinatorics.graycode import GrayCode


# make the graph (hierarchy)
class Graph(object):
    def __init__(self, num_nodes):
        self.adjacency_matrix = []
        for i in range(num_nodes):
            self.adjacency_matrix.append([0 for i in range(num_nodes)])
        self.numNodes = num_nodes

    def add_edge(self, start, end):
        self.adjacency_matrix[start][end] = 1

    def remove_edge(self, start, end):
        if self.adjacency_matrix[start][end] == 0:
            print("There is no edge between %d and %d" % (start, end))
        else:
            self.adjacency_matrix[start][end] = 0

    def contains_edge(self, start, end):
        if self.adjacency_matrix[start][end] > 0:
            return True
        else:
            return False

    def __len__(self):
        return self.numNodes


def make_all_concept_state_mat(num_of_concepts):
    """
    generate 2^n x n matrix of all concept states
    :param num_of_concepts:
    :return:
    """
    concept_state = list()
    all_concept_state = sorted(list(GrayCode(num_of_concepts).generate_gray()))
    
    for state in all_concept_state:
        for i in state:
            concept_state.append(int(i))
            
    concept_state = np.reshape(concept_state,(-1,num_of_concepts))
    
    return concept_state


def calc_reachability(A):
    """
    calc reachability matrix R from adjacency matrix A
    R_ij = 1 means there exists a path from node i to node j
    :param A: adjacency matrix
    :return: reachability matrix R
    """
    if isinstance(A, list):
        A = np.array(A)

    num_attribute = A.shape[0]
    I = np.identity(num_attribute)

    # repeat multiplications
    R = A + I
    # print(R)
    for i in range(num_attribute):
        R = np.dot(R, A + I)
        # print(R)

    R = (R > 0).astype(int)   # binarize (Don't care no. of visits.)
    # for i in range(R.shape[0]):
    #     for j in range(R.shape[1]):
    #         if R[i][j] >= 1:
    #             R[i][j] = 1
    #         else:
    #             continue
    return R


def calc_knowledge_states(A, reduce=True):
    num_attribute = A.shape[1]


    # start from all possible states
    S = make_all_concept_state_mat(num_attribute)
    # print(S.shape)

    R = calc_reachability(A)

    for item in range(S.shape[0]):
        for attribute in range(S.shape[1]):
            if S[item,attribute] == 1:
                for i in np.where(R[:, attribute] == 1)[0]:   # fill 1 for all the "preceding" attribute
                    S[item,i] = 1
    # print(S.shape)

    # reduce S
    if reduce:
        S = np.unique(S, axis=0)
    # print(S.shape)

    return S


def calc_Q(R, reduce=True):
    num_attribute = R.shape[0]

    # start from all possible states
    Q = make_all_concept_state_mat(num_attribute)[1:]
    # print(Q.shape)

    for item in range(Q.shape[0]):
        for attribute in range(Q.shape[1]):
            if Q[item,attribute] == 1:
                for i in np.where(R[:, attribute] == 1)[0]:   # fill 1 for all the "preceding" attribute
                    Q[item,i] = 1
    # print(Q.shape)

    # reduce Q
    if reduce:
        Q = np.unique(Q, axis=0)
    # print(Q.shape)

    return Q


def calc_ideal_responses(Q, concept_state=None):
    num_attribute = Q.shape[1]
    if concept_state is None:
        concept_state = make_all_concept_state_mat(num_attribute)

    ideal_responses = sorted(list(set(map(tuple, make_ideal_response(concept_state, Q.T)))))

    return ideal_responses


def NOT_GATE(input_vectors):
    result = list()
    for i in input_vectors.tolist():
        for j in range(len(i)):
            if i[j] == 0:
                i[j] = 1
            else:
                i[j] = 0      
        result.append(i)
    return result


# ideal response (Barnes, 2005)
def make_ideal_response(concept_state, Q):
    tmp = np.dot(NOT_GATE(concept_state), Q)
    for i in tmp:
        for j in range(len(i)):
            if i[j] == 0:
                pass
            else:
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
    
    return mapping_index.astype(int)


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
