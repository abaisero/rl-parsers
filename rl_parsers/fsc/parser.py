from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np

from ..errors import SemanticError
from . import tokrules


@dataclass
class FSC:
    nodes: Union[List[str], List[int]]
    actions: Union[List[str], List[int]]
    observations: Union[List[str], List[int]]

    start: np.ndarray
    A: np.ndarray
    T: np.ndarray

    flags: Dict[str, Any]


class Parser:  # pylint: disable=too-many-public-methods
    tokens = tokrules.tokens

    def __init__(self):
        self.nodes = None
        self.actions = None
        self.observations = None

        self.num_nodes = None
        self.num_actions = None
        self.num_observations = None

        self.start = None
        self.A = None
        self.T = None

        self.flags = {}

    def p_error(self, p):  # pylint: disable=no-self-use
        print(
            f'Parsing Error line={p.lineno} pos={p.lexpos} type={p.type} value={p.value}'
        )

    def p_fsc(self, p):
        """ fsc : preamble start structure
                | preamble structure """

        p[0] = FSC(
            nodes=self.nodes,
            actions=self.actions,
            observations=self.observations,
            start=self.start,
            A=self.A,
            T=self.T,
            flags=self.flags,
        )

    ### PREAMBLE

    def p_preamble(self, p):  # pylint: disable=unused-argument
        """ preamble : preamble_list """
        self.A = np.zeros((self.num_nodes, self.num_actions))
        self.T = np.zeros(
            (self.num_observations, self.num_nodes, self.num_nodes)
        )

    def p_preamble_list(self, p):
        """ preamble_list : preamble_list preamble_item
                          | preamble_item """

    def p_preamble_nodes_N(self, p):
        """ preamble_item : NODES COLON INT """
        N = p[3]
        self.nodes = list(range(N))
        self.num_nodes = N

    def p_preamble_nodes_names(self, p):
        """ preamble_item : NODES COLON id_list """
        idlist = p[3]
        self.nodes = list(idlist)
        self.num_nodes = len(idlist)

    def p_preamble_actions_N(self, p):
        """ preamble_item : ACTIONS COLON INT """
        N = p[3]
        self.actions = list(range(N))
        self.num_actions = N

    def p_preamble_actions_names(self, p):
        """ preamble_item : ACTIONS COLON id_list """
        idlist = p[3]
        self.actions = list(idlist)
        self.num_actions = len(idlist)

    def p_preamble_observations_N(self, p):
        """ preamble_item : OBSERVATIONS COLON INT """
        N = p[3]
        self.observations = list(range(N))
        self.num_observations = N

    def p_preamble_observations_names(self, p):
        """ preamble_item : OBSERVATIONS COLON id_list """
        idlist = p[3]
        self.observations = list(idlist)
        self.num_observations = len(idlist)

    ### START

    def p_start_uniform(self, p):  # pylint: disable=no-self-use,unused-argument
        """ start : START COLON UNIFORM """
        self.start = np.full(self.num_nodes, 1 / self.num_nodes)

    # NOTE reduce/reduce conflict solved by enforcing pmatrix contains at least
    # 2 probabilities
    def p_start_dist(self, p):
        """ start : START COLON pmatrix """
        pm = np.array(p[3])
        if not np.isclose(pm.sum(), 1.0):
            raise SemanticError(
                f'Start distribution is not normalized (sums to {pm.sum()}).'
            )
        self.start = pm

    def p_start_node(self, p):
        """ start : START COLON node """
        s = p[3]
        self.start = np.zeros(self.num_nodes)
        self.start[s] = 1

    def p_start_include(self, p):
        """ start : START INCLUDE COLON node_list """
        node_list = p[4]
        self.start = np.zeros(self.num_nodes)
        self.start[node_list] = 1 / len(node_list)

    def p_start_exclude(self, p):
        """ start : START EXCLUDE COLON node_list """
        node_list = p[4]
        self.start = np.full(
            self.num_nodes, 1 / (self.num_nodes - len(node_list))
        )
        self.start[node_list] = 0

    ### ID LIST

    def p_id_list(self, p):  # pylint: disable=no-self-use
        """ id_list : id_list ID """
        p[0] = p[1] + [p[2]]

    def p_id_list_base(self, p):  # pylint: disable=no-self-use
        """ id_list : ID """
        p[0] = [p[1]]

    ### NODE LIST

    def p_node_list(self, p):  # pylint: disable=no-self-use
        """ node_list : node_list node """
        p[0] = p[1] + [p[2]]

    def p_node_list_base(self, p):  # pylint: disable=no-self-use
        """ node_list : node """
        p[0] = [p[1]]

    ### NODE

    def p_node_idx(self, p):  # pylint: disable=no-self-use
        """ node : INT """
        p[0] = p[1]

    def p_node_id(self, p):
        """ node : ID """
        p[0] = self.nodes.index(p[1])

    def p_node_all(self, p):  # pylint: disable=no-self-use
        """ node : ASTERISK """
        p[0] = slice(None)

    ### ACTION

    def p_action_idx(self, p):  # pylint: disable=no-self-use
        """ action : INT """
        p[0] = p[1]

    def p_action_id(self, p):
        """ action : ID """
        p[0] = self.actions.index(p[1])

    def p_action_all(self, p):  # pylint: disable=no-self-use
        """ action : ASTERISK """
        p[0] = slice(None)

    ### OBSERVATION

    def p_observation_idx(self, p):  # pylint: disable=no-self-use
        """ observation : INT """
        p[0] = p[1]

    def p_observation_id(self, p):
        """ observation : ID """
        p[0] = self.observations.index(p[1])

    def p_observation_all(self, p):  # pylint: disable=no-self-use
        """ observation : ASTERISK """
        p[0] = slice(None)

    ### STRUCTURE

    def p_structure(self, p):
        """ structure : structure_list """

    def p_structure_list(self, p):
        """ structure_list : structure_list structure_item
                           | """

    ### STRUCTURE A

    def p_structure_a_na(self, p):
        """ structure_item : A COLON node COLON action prob """
        n0, a, prob = p[3], p[5], p[6]
        self.A[n0, a] = prob

    def p_structure_a_n_uniform(self, p):
        """ structure_item : A COLON node UNIFORM """
        n0 = p[3]
        self.T[n0] = 1 / self.num_nodes

    def p_structure_a_n_dist(self, p):
        """ structure_item : A COLON node pmatrix """
        n0, pm = p[3], p[4]
        pm = np.array(pm)
        if not np.isclose(pm.sum(), 1.0):
            raise SemanticError(
                f'Action distribution (node={n0}) is not normalized (sums to {pm.sum()}).'
            )
        self.A[n0] = pm

    ### STRUCTURE T

    def p_structure_t_ass(self, p):
        """ structure_item : T COLON observation COLON node COLON node prob """
        o, n0, n1, prob = p[3], p[5], p[7], p[8]
        self.T[o, n0, n1] = prob

    def p_structure_t_as_uniform(self, p):
        """ structure_item : T COLON observation COLON node UNIFORM """
        o, n0 = p[3], p[5]
        self.T[o, n0] = 1 / self.num_nodes

    def p_structure_t_os_reset(self, p):
        """ structure_item : T COLON observation COLON node RESET """
        o, n0 = p[3], p[5]
        self.T[o, n0] = self.start

    def p_structure_t_os_dist(self, p):
        """ structure_item : T COLON observation COLON node pmatrix """
        o, n0, pm = p[3], p[5], p[6]
        pm = np.array(pm)
        if not np.isclose(pm.sum(), 1.0):
            raise SemanticError(
                f'Transition distribution (observation={o}, ode={n0}) is not normalized (sums to {pm.sum()}).'  # pylint: disable=line-too-long
            )
        self.T[o, n0] = pm

    def p_structure_t_o_uniform(self, p):
        """ structure_item : T COLON observation UNIFORM """
        o = p[3]
        self.T[o] = 1 / self.num_nodes

    def p_structure_t_o_identity(self, p):
        """ structure_item : T COLON observation IDENTITY """
        o = p[3]
        self.T[o] = np.eye(self.num_nodes)

    def p_structure_t_o_dist(self, p):
        """ structure_item : T COLON observation pmatrix """
        o, pm = p[3], p[4]
        pm = np.reshape(pm, (self.num_nodes, self.num_nodes))
        if not np.isclose(pm.sum(axis=1), 1.0).all():
            raise SemanticError(
                f'Transition node distribution (observation={o}) is not normalized;'
            )
        self.T[o] = pm

    ### PMATRIX

    def p_pmatrix(self, p):  # pylint: disable=no-self-use
        """ pmatrix : pmatrix prob """
        p[0] = p[1] + [p[2]]

    # NOTE enforcing 2 probabilities solves the reduce/reduce conflict in
    # start_node rule
    def p_pmatrix_base(self, p):  # pylint: disable=no-self-use
        """ pmatrix : prob prob """
        p[0] = [p[1], p[2]]

    ### PROB

    def p_prob(self, p):  # pylint: disable=no-self-use
        """ prob : FLOAT
                 | INT """
        prob = float(p[1])

        if not 0 <= prob <= 1:
            raise SemanticError(f'Probability value ({prob}) out of bounds.')

        p[0] = prob
