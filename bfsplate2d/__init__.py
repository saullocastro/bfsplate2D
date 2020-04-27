"""
BFSPLATE2D - Implementation of the BFS plate finite element in 2D

Author: Saullo G. P. Castro

"""
from .bfsplate2d import BFSPlate2D, update_KC0, update_KG, update_M, update_KA
from .bfsplate2d import INT, DOUBLE, KC0_SPARSE_SIZE, KG_SPARSE_SIZE, M_SPARSE_SIZE, KA_SPARSE_SIZE
DOF = 6

