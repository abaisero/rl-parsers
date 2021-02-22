from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np

from ..errors import SemanticError
from . import tokrules


@dataclass
class POMDP:
    discount: float
    values: str

    states: Union[List[str], List[int]]
    actions: Union[List[str], List[int]]
    observations: Union[List[str], List[int]]

    start: np.ndarray
    T: np.ndarray
    O: np.ndarray
    R: np.ndarray

    reset: np.ndarray
    flags: Dict[str, Any]


class Parser:  # pylint: disable=too-many-public-methods
    tokens = tokrules.tokens

    def __init__(self):
        self.discount = None
        self.values = None

        self.states = None
        self.actions = None
        self.observations = None

        self.num_states = None
        self.num_actions = None
        self.num_observations = None

        self.start = None
        self.T = None
        self.O = None
        self.R = None

        self.reset = None

        self.flags = {}

    def _extend_O(self):
        if not self.flags['O_includes_state']:
            self.flags['O_includes_state'] = True
            self.O = np.expand_dims(self.O, axis=1).repeat(
                self.num_states, axis=1
            )

    def p_error(self, p):  # pylint: disable=no-self-use
        print(
            f'Parsing Error line={p.lineno} pos={p.lexpos} type={p.type} value={p.value}'
        )

    def p_pomdp(self, p):
        """ pomdp : preamble start structure
                  | preamble structure """

        not_close_mask = ~np.isclose(self.T.sum(-1), 1.0)
        if not_close_mask.any():
            not_close_states, not_close_actions = not_close_mask.nonzero()
            a_idx, s_idx = not_close_actions[0], not_close_states[0]
            raise SemanticError(
                f'T[{self.actions[a_idx]}, {self.states[s_idx]}] = {self.T[a_idx, s_idx]} (and {not_close_mask.sum() - 1} other transitions) does not add up to 1.0'  # pylint: disable=line-too-long
            )

        not_close_mask = ~np.isclose(self.O.sum(-1), 1.0)
        if not_close_mask.any():
            not_close_states, not_close_actions = not_close_mask.nonzero()
            a_idx, s_idx = not_close_actions[0], not_close_states[0]
            raise SemanticError(
                f'O[{self.actions[a_idx]}, {self.states[s_idx]}] = {self.O[a_idx, s_idx]} (and {not_close_mask.sum() - 1} other transitions) does not add up to 1.0'  # pylint: disable=line-too-long
            )

        p[0] = POMDP(
            discount=self.discount,
            values=self.values,
            states=self.states,
            actions=self.actions,
            observations=self.observations,
            start=self.start,
            T=self.T,
            O=self.O,
            R=self.R,
            reset=self.reset,
            flags=self.flags,
        )

    ### PREAMBLE

    def p_preamble(self, p):  # pylint: disable=no-self-use,unused-argument
        """ preamble : preamble_list """
        self.T = np.zeros((self.num_actions, self.num_states, self.num_states))
        self.O = np.zeros(
            (self.num_actions, self.num_states, self.num_observations)
        )
        self.R = np.zeros(
            (
                self.num_actions,
                self.num_states,
                self.num_states,
                self.num_observations,
            )
        )
        self.reset = np.zeros(
            (self.num_actions, self.num_states), dtype=bool
        )

        self.flags['O_includes_state'] = False

    def p_preamble_list(self, p):
        """ preamble_list : preamble_list preamble_item
                          | preamble_item """

    def p_preamble_discount(self, p):  # pylint: disable=no-self-use
        """ preamble_item : DISCOUNT COLON number """
        discount = p[3]

        if not 0 <= discount <= 1:
            raise SemanticError(f'Discount value {discount} out of bounds.')

        self.discount = discount

    def p_preamble_values(self, p):
        """ preamble_item : VALUES COLON REWARD
                          | VALUES COLON REWARDS
                          | VALUES COLON COST
                          | VALUES COLON COSTS """
        self.values = p[3]

    def p_preamble_states_N(self, p):
        """ preamble_item : STATES COLON INT """
        N = p[3]
        self.states = list(range(N))
        self.num_states = len(self.states)

    def p_preamble_states_names(self, p):
        """ preamble_item : STATES COLON id_list """
        idlist = p[3]
        self.states = list(idlist)
        self.num_states = len(self.states)

    def p_preamble_actions_N(self, p):
        """ preamble_item : ACTIONS COLON INT """
        N = p[3]
        self.actions = list(range(N))
        self.num_actions = len(self.actions)

    def p_preamble_actions_names(self, p):
        """ preamble_item : ACTIONS COLON id_list """
        idlist = p[3]
        self.actions = list(idlist)
        self.num_actions = len(self.actions)

    def p_preamble_observations_N(self, p):
        """ preamble_item : OBSERVATIONS COLON INT """
        N = p[3]
        self.observations = list(range(N))
        self.num_observations = len(self.observations)

    def p_preamble_observations_names(self, p):
        """ preamble_item : OBSERVATIONS COLON id_list """
        idlist = p[3]
        self.observations = list(idlist)
        self.num_observations = len(self.observations)

    ### START

    def p_start_uniform(  # pylint: disable=unused-argument,unused-argument
        self, p
    ):
        """ start : START COLON UNIFORM """
        self.start = np.full(self.num_states, 1 / self.num_states)

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

    def p_start_state(self, p):
        """ start : START COLON state """
        s = p[3]
        self.start = np.zeros(self.num_states)
        self.start[s] = 1

    def p_start_include(self, p):
        """ start : START INCLUDE COLON state_list """
        state_list = p[4]
        self.start = np.zeros(self.num_states)
        self.start[state_list] = 1 / len(state_list)

    def p_start_exclude(self, p):
        """ start : START EXCLUDE COLON state_list """
        state_list = p[4]
        self.start = np.full(
            self.num_states, 1 / (self.num_states - len(state_list))
        )
        self.start[state_list] = 0

    ### ID LIST

    def p_id_list(self, p):  # pylint: disable=no-self-use
        """ id_list : id_list ID """
        p[0] = p[1] + [p[2]]

    def p_id_list_base(self, p):  # pylint: disable=no-self-use
        """ id_list : ID """
        p[0] = [p[1]]

    ### STATE LIST

    def p_state_list(self, p):  # pylint: disable=no-self-use
        """ state_list : state_list state """
        p[0] = p[1] + [p[2]]

    def p_state_list_base(self, p):  # pylint: disable=no-self-use
        """ state_list : state """
        p[0] = [p[1]]

    ### STATE

    def p_state_idx(self, p):  # pylint: disable=no-self-use
        """ state : INT """
        p[0] = p[1]

    def p_state_id(self, p):
        """ state : ID """
        p[0] = self.states.index(p[1])

    def p_state_all(self, p):  # pylint: disable=no-self-use
        """ state : ASTERISK """
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

    def p_structure_t_ass(self, p):
        """ structure_item : T COLON action COLON state COLON state prob """
        a, s0, s1, prob = p[3], p[5], p[7], p[8]
        self.T[a, s0, s1] = prob

    def p_structure_t_as_uniform(self, p):
        """ structure_item : T COLON action COLON state UNIFORM """
        a, s0 = p[3], p[5]
        self.T[a, s0] = 1 / self.num_states

    def p_structure_t_as_reset(self, p):
        """ structure_item : T COLON action COLON state RESET """
        a, s0 = p[3], p[5]
        self.T[a, s0] = self.start
        self.reset[a, s0] = True

    def p_structure_t_as_dist(self, p):
        """ structure_item : T COLON action COLON state pmatrix """
        a, s0, pm = p[3], p[5], p[6]
        pm = np.array(pm)
        if not np.isclose(pm.sum(), 1.0):
            raise SemanticError(
                f'Transition distribution (action={a}, state={s0}) is not normalized (sums to {pm.sum()}).'  # pylint: disable=line-too-long
            )

        self.T[a, s0] = pm

    def p_structure_t_a_uniform(self, p):
        """ structure_item : T COLON action UNIFORM """
        a = p[3]
        self.T[a] = 1 / self.num_states

    def p_structure_t_a_identity(self, p):
        """ structure_item : T COLON action IDENTITY """
        a = p[3]
        self.T[a] = np.eye(self.num_states)

    def p_structure_t_a_dist(self, p):
        """ structure_item : T COLON action pmatrix """
        a, pm = p[3], p[4]
        pm = np.reshape(pm, (self.num_states, self.num_states))
        if not np.isclose(pm.sum(axis=1), 1.0).all():
            raise SemanticError(
                f'Transition state distribution (action={a}) is not normalized;'
            )
        self.T[a] = pm

    ### STRUCTURE O

    def p_structure_o_aso(self, p):
        """ structure_item : O COLON action COLON state COLON observation prob """
        a, s1, o, pr = p[3], p[5], p[7], p[8]
        if self.flags['O_includes_state']:
            self.O[a, :, s1, o] = pr
        else:
            self.O[a, s1, o] = pr

    def p_structure_o_as_uniform(self, p):
        """ structure_item : O COLON action COLON state UNIFORM """
        a, s1 = p[3], p[5]
        if self.flags['O_includes_state']:
            self.O[a, :, s1] = 1 / self.num_observations
        else:
            self.O[a, s1] = 1 / self.num_observations

    def p_structure_o_as_dist(self, p):
        """ structure_item : O COLON action COLON state pmatrix """
        a, s1, pm = p[3], p[5], p[6]
        if self.flags['O_includes_state']:
            self.O[a, :, s1] = pm
        else:
            self.O[a, s1] = pm

    def p_structure_o_a_uniform(self, p):
        """ structure_item : O COLON action UNIFORM """
        a = p[3]
        if self.flags['O_includes_state']:
            self.O[a, :] = 1 / self.num_observations
        else:
            self.O[a] = 1 / self.num_observations

    def p_structure_o_a_dist(self, p):
        """ structure_item : O COLON action pmatrix """
        a, pm = p[3], p[4]
        if self.flags['O_includes_state']:
            self.O[a, :] = np.reshape(
                pm, (self.num_states, self.num_observations)
            )
        else:
            self.O[a] = np.reshape(pm, (self.num_states, self.num_observations))

    ### STRUCTURE OO

    def p_structure_oo_asso(self, p):
        """ structure_item : OO COLON action COLON state COLON state COLON observation prob """
        a, s0, s1, o, pr = p[3], p[5], p[7], p[9], p[10]
        self._extend_O()
        self.O[a, s0, s1, o] = pr

    def p_structure_oo_ass_uniform(self, p):
        """ structure_item : OO COLON action COLON state COLON state UNIFORM """
        a, s0, s1 = p[3], p[5], p[7]
        self._extend_O()
        self.O[a, s0, s1] = 1 / self.num_observations

    def p_structure_oo_ass_dist(self, p):
        """ structure_item : OO COLON action COLON state COLON state pmatrix """
        a, s0, s1, pm = p[3], p[5], p[7], p[8]
        self._extend_O()
        self.O[a, s0, s1] = pm

    def p_structure_oo_as_uniform(self, p):
        """ structure_item : OO COLON action COLON state UNIFORM """
        a, s0 = p[3], p[5]
        self._extend_O()
        self.O[a, s0] = 1 / self.num_observations

    def p_structure_oo_as_dist(self, p):
        """ structure_item : OO COLON action COLON state pmatrix """
        a, s0, pm = p[3], p[5], p[6]
        self._extend_O()
        self.O[a, s0] = np.reshape(pm, (self.num_states, self.num_observations))

    ### STRUCTURE R

    def p_structure_r_asso(self, p):
        """ structure_item : R COLON action COLON state COLON state COLON observation number """
        a, s0, s1, o, r = p[3], p[5], p[7], p[9], p[10]
        self.R[a, s0, s1, o] = r

    def p_structure_r_ass(self, p):
        """ structure_item : R COLON action COLON state COLON state nmatrix """
        a, s0, s1, r = p[3], p[5], p[7], p[8]
        self.R[a, s0, s1] = r

    def p_structure_r_as(self, p):
        """ structure_item : R COLON action COLON state nmatrix """
        a, s0, r = p[3], p[5], p[6]
        self.R[a, s0] = r

    ### PMATRIX

    def p_pmatrix(self, p):  # pylint: disable=no-self-use
        """ pmatrix : pmatrix prob """
        p[0] = p[1] + [p[2]]

    # NOTE enforcing 2 probabilities solves the reduce/reduce conflict in
    # start_state rule
    def p_pmatrix_base(self, p):  # pylint: disable=no-self-use
        """ pmatrix : prob prob """
        p[0] = [p[1], p[2]]

    ### NMATRIX

    def p_nmatrix(self, p):  # pylint: disable=no-self-use
        """ nmatrix : nmatrix number """
        p[0] = p[1] + [p[2]]

    def p_nmatrix_base(self, p):  # pylint: disable=no-self-use
        """ nmatrix : number """
        p[0] = [p[1]]

    ### NUMBER

    def p_number(self, p):  # pylint: disable=no-self-use
        """ number : number_nosign
                   | number_plus
                   | number_minus """
        p[0] = p[1]

    def p_number_nosign(self, p):  # pylint: disable=no-self-use
        """ number_nosign : FLOAT
                          | INT """
        p[0] = float(p[1])

    def p_number_plus(self, p):  # pylint: disable=no-self-use
        """ number_plus : PLUS FLOAT
                        | PLUS INT """
        p[0] = float(p[2])

    def p_number_minus(self, p):  # pylint: disable=no-self-use
        """ number_minus : MINUS FLOAT
                         | MINUS INT """
        p[0] = -float(p[2])

    ### PROB

    def p_prob(self, p):  # pylint: disable=no-self-use
        """ prob : FLOAT
                 | INT """
        prob = float(p[1])

        if not 0 <= prob <= 1:
            raise SemanticError(f'Probability value {prob} out of bounds.')

        p[0] = prob
