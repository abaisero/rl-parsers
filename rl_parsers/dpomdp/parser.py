from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np

from ..errors import SemanticError
from . import tokrules


@dataclass
class DPOMDP:
    discount: float
    values: str

    agents: Union[List[str], List[int]]
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

        self.agents = None
        self.states = None
        self.actions = None
        self.observations = None

        self.num_agents = None
        self.num_states = None
        self.num_actions = None
        self.num_observations = None

        self.action_strides = None

        self.start = None
        self.T = None
        self.O = None
        self.R = None

        self.reset = None

        self.flags = {}

    def p_error(self, p):  # pylint: disable=no-self-use
        print(
            f'Parsing Error line={p.lineno} pos={p.lexpos} type={p.type} value={p.value}'
        )

    def p_decpomdp(self, p):
        """ decpomdp : preamble structure """

        not_close_mask = ~np.isclose(self.T.sum(-1), 1.0)
        if not_close_mask.any():
            not_close_states, not_close_actions = not_close_mask.nonzero()
            a_idx, s_idx = not_close_actions[0], not_close_states[0]
            raise SemanticError(
                f'T[{self.actions[a_idx]}, {self.states[s_idx]}] = {self.T[a_idx, s_idx]} (and {not_close_mask.sum() - 1} other transitions) does not add up to 1.0'  # pylint: disable=line-too-long
            )

        sum_indices = (*range(-len(self.num_observations), 0),)
        not_close_mask = ~np.isclose(self.O.sum(sum_indices), 1.0)
        if not_close_mask.any():
            not_close_states, not_close_actions = not_close_mask.nonzero()
            a_idx, s_idx = not_close_actions[0], not_close_states[0]
            raise SemanticError(
                f'O[{self.actions[a_idx]}, {self.states[s_idx]}] = {self.O[a_idx, s_idx]} (and {not_close_mask.sum() - 1} other transitions) does not add up to 1.0'  # pylint: disable=line-too-long
            )

        p[0] = DPOMDP(
            discount=self.discount,
            values=self.values,
            agents=self.agents,
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
        self.T = np.zeros((*self.num_actions, self.num_states, self.num_states))
        self.O = np.zeros(
            (*self.num_actions, self.num_states, *self.num_observations)
        )
        self.R = np.zeros(
            (
                *self.num_actions,
                self.num_states,
                self.num_states,
                *self.num_observations,
            )
        )
        self.reset = np.zeros((*self.num_actions, self.num_states), dtype=bool)

    def p_preamble_list(self, p):
        """ preamble_list : preamble_list preamble_item
                          | preamble_item
                          | NL """

    def p_preamble_discount(self, p):  # pylint: disable=no-self-use
        """ preamble_item : DISCOUNT COLON number NL """
        discount = p[3]

        if not 0 <= discount <= 1:
            raise SemanticError(f'Discount value {discount} out of bounds.')

        self.discount = discount

    def p_preamble_values(self, p):
        """ preamble_item : VALUES COLON REWARD NL
                          | VALUES COLON REWARDS NL
                          | VALUES COLON COST NL
                          | VALUES COLON COSTS NL """
        self.values = p[3]

    def p_preamble_agents_N(self, p):
        """ preamble_item : AGENTS COLON INT NL """
        N = p[3]
        self.agents = list(range(N))
        self.num_agents = len(self.agents)

    def p_preamble_agents_names(self, p):
        """ preamble_item : AGENTS COLON id_list NL """
        idlist = p[3]
        self.agents = list(idlist)
        self.num_agents = len(self.agents)

    def p_preamble_states_N(self, p):
        """ preamble_item : STATES COLON INT NL """
        N = p[3]
        self.states = list(range(N))
        self.num_states = len(self.states)

    def p_preamble_states_names(self, p):
        """ preamble_item : STATES COLON id_list NL """
        idlist = p[3]
        self.states = list(idlist)
        self.num_states = len(self.states)

    def p_preamble_actions(self, p):
        """ preamble_item : ACTIONS COLON NL joint_list """
        joint_list = p[4]
        self.actions = list(joint_list)
        self.num_actions = list(map(len, joint_list))
        self.action_strides = np.cumprod(self.num_actions[::-1])[::-1]

    def p_preamble_observations(self, p):
        """ preamble_item : OBSERVATIONS COLON NL joint_list """
        joint_list = p[4]
        self.observations = list(joint_list)
        self.num_observations = list(map(len, joint_list))

    ###

    def p_preamble_joint_list(self, p):  # pylint: disable=no-self-use
        """ joint_list : joint_list joint_item """
        p[0] = p[1] + [p[2]]

    def p_preamble_joint_base(self, p):  # pylint: disable=no-self-use
        """ joint_list : joint_item """
        p[0] = [p[1]]

    def p_preamble_joint_N(self, p):  # pylint: disable=no-self-use
        """ joint_item : INT NL """
        p[0] = list(range(p[1]))

    def p_preamble_joint_names(self, p):  # pylint: disable=no-self-use
        """ joint_item : id_list NL """
        p[0] = list(p[1])

    ### START

    def p_preamble_start_uniform(self, p):  # pylint: disable=unused-argument
        """ preamble_item : START COLON NL UNIFORM NL """
        self.start = np.full(self.num_states, 1 / self.num_states)

    # NOTE reduce/reduce conflict solved by enforcing pmatrix contains at least
    # 2 probabilities
    def p_preamble_start_dist(self, p):
        """ preamble_item : START COLON NL pvector NL """
        pv = np.array(p[4])
        if not np.isclose(pv.sum(), 1.0):
            raise SemanticError(
                f'Start distribution is not normalized (sums to {pv.sum()}).'
            )
        self.start = pv

    def p_preamble_start_state(self, p):
        """ preamble_item : START COLON state NL """
        s = p[3]
        self.start = np.zeros(self.num_states)
        self.start[s] = 1

    def p_preamble_start_include(self, p):
        """ preamble_item : START INCLUDE COLON NL state_list NL """
        state_list = p[5]
        self.start = np.zeros(self.num_states)
        self.start[state_list] = 1 / len(state_list)

    def p_preamble_start_exclude(self, p):
        """ preamble_item : START EXCLUDE COLON NL state_list NL """
        state_list = p[5]
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

    ### JOINT ACTION

    def p_jaction(self, p):
        """ jaction : action_list """
        actions = p[1]

        if len(actions) == 1:
            action = actions[0]
            if isinstance(action, int):
                jaction = list(action // self.action_strides)
            elif action == slice(None):
                jaction = [slice(None)] * self.num_agents
        elif len(actions) == self.num_agents:
            jaction = list(actions)
            for i, action in enumerate(actions):
                if isinstance(action, str):
                    jaction[i] = self.actions[i].index(action)
        else:
            raise SemanticError(
                'Joint action should contain either one or enough indices for each agent'
            )

        p[0] = list(jaction)

    def p_action_list(self, p):  # pylint: disable=no-self-use
        """ action_list : action_list action """
        p[0] = p[1] + [p[2]]

    def p_jactions_base(self, p):  # pylint: disable=no-self-use
        """ action_list : action """
        p[0] = [p[1]]

    def p_action(self, p):  # pylint: disable=no-self-use
        """ action : INT
                   | ID """
        p[0] = p[1]

    def p_action_asterisk(self, p):  # pylint: disable=no-self-use
        """ action : ASTERISK """
        p[0] = slice(None)

    ### OBSERVATION

    def p_jobservation(self, p):
        """ jobservation : observation_list """
        observations = p[1]

        if len(observations) == 1 and observations[0] == slice(None):
            jobservation = [slice(None)] * self.num_agents
        elif len(observations) == self.num_agents:
            jobservation = list(observations)
            for i, observation in enumerate(observations):
                if isinstance(observation, str):
                    jobservation[i] = self.observations[i].index(observation)
        else:
            raise SemanticError(
                'Joint observation should contain either one or enough indices for each agent'
            )

        p[0] = list(jobservation)

    def p_observation_list(self, p):  # pylint: disable=no-self-use
        """ observation_list : observation_list observation """
        p[0] = p[1] + [p[2]]

    def p_jobservations_base(self, p):  # pylint: disable=no-self-use
        """ observation_list : observation """
        p[0] = [p[1]]

    def p_observation(self, p):  # pylint: disable=no-self-use
        """ observation : INT
                        | ID """
        p[0] = p[1]

    def p_observation_asterisk(self, p):  # pylint: disable=no-self-use
        """ observation : ASTERISK """
        p[0] = slice(None)

    ### STRUCTURE

    def p_structure(self, p):
        """ structure : structure_list """

    def p_structure_list(self, p):
        """ structure_list : structure_list structure_item
                           | """

    def p_structure_t_ass(self, p):
        """ structure_item : T COLON jaction COLON state COLON state COLON prob NL """
        ja, s0, s1, prob = p[3], p[5], p[7], p[9]
        self.T[(*ja, s0, s1)] = prob

    def p_structure_t_as_uniform(self, p):
        """ structure_item : T COLON jaction COLON state COLON NL UNIFORM NL """
        ja, s0 = p[3], p[5]
        self.T[(*ja, s0)] = 1 / self.num_states

    def p_structure_t_as_reset(self, p):
        """ structure_item : T COLON jaction COLON state COLON NL RESET NL """
        ja, s0 = p[3], p[5]
        self.T[(*ja, s0)] = self.start
        self.reset[(*ja, s0)] = True

    def p_structure_t_as_dist(self, p):
        """ structure_item : T COLON jaction COLON state COLON NL pvector NL """
        ja, s0, pv = p[3], p[5], p[8]
        pv = np.array(pv)
        if not np.isclose(pv.sum(), 1.0):
            raise SemanticError(
                f'Transition distribution (action={ja}, state={s0}) is not normalized (sums to {pv.sum()}).'  # pylint: disable=line-too-long
            )

        self.T[(*ja, s0)] = pv

    def p_structure_t_a_uniform(self, p):
        """ structure_item : T COLON jaction COLON NL UNIFORM NL """
        ja = p[3]
        self.T[(*ja,)] = 1 / self.num_states

    def p_structure_t_a_identity(self, p):
        """ structure_item : T COLON jaction COLON NL IDENTITY NL """
        ja = p[3]
        self.T[(*ja,)] = np.eye(self.num_states)

    def p_structure_t_a_dist(self, p):
        """ structure_item : T COLON jaction COLON NL pmatrix NL """
        ja, pm = p[3], p[5]
        pm = np.reshape(pm, (self.num_states, self.num_states))
        if not np.isclose(pm.sum(axis=1), 1.0).all():
            raise SemanticError(
                f'Transition state distribution (action={ja}) is not normalized'
            )
        self.T[(*ja,)] = pm

    ### STRUCTURE O

    def p_structure_o_aso(self, p):
        """ structure_item : O COLON jaction COLON state COLON jobservation COLON prob NL """
        ja, s1, jo, pr = p[3], p[5], p[7], p[9]
        self.O[(*ja, s1, *jo)] = pr

    def p_structure_o_as_uniform(self, p):
        """ structure_item : O COLON jaction COLON state COLON NL UNIFORM NL """
        ja, s1 = p[3], p[5]
        self.O[(*ja, s1)] = 1 / np.prod(self.num_observations)

    def p_structure_o_as_dist(self, p):
        """ structure_item : O COLON jaction COLON state COLON NL pvector NL """
        ja, s1, pv = p[3], p[5], p[7]
        self.O[(*ja, s1)] = np.reshape(pv, self.num_observations)

    def p_structure_o_a_uniform(self, p):
        """ structure_item : O COLON jaction COLON NL UNIFORM NL """
        ja = p[3]
        self.O[(*ja,)] = 1 / np.prod(self.num_observations)

    def p_structure_o_a_dist(self, p):
        """ structure_item : O COLON jaction COLON NL pmatrix NL """
        ja, pm = p[3], p[5]
        self.O[(*ja,)] = np.reshape(
            pm, (self.num_states, self.num_observations)
        )

    ### STRUCTURE R

    def p_structure_r_asso(self, p):
        """ structure_item : R COLON jaction COLON state COLON state COLON jobservation COLON number NL """
        ja, s0, s1, jo, r = p[3], p[5], p[7], p[9], p[11]
        self.R[(*ja, s0, s1, *jo)] = r

    def p_structure_r_ass(self, p):
        """ structure_item : R COLON jaction COLON state COLON state COLON NL nvector NL """
        ja, s0, s1, rv = p[3], p[5], p[7], p[10]
        self.R[(*ja, s0, s1)] = rv

    def p_structure_r_as(self, p):
        """ structure_item : R COLON jaction COLON state nmatrix """
        ja, s0, rm = p[3], p[5], p[6]
        self.R[(*ja, s0)] = rm

    ### PROBABILITIES

    def p_pmatrix(self, p):  # pylint: disable=no-self-use
        """ pmatrix : pmatrix NL pvector """
        p[0] = p[1] + [p[3]]

    def p_pmatrix_base(self, p):  # pylint: disable=no-self-use
        """ pmatrix : pvector """
        p[0] = [p[1]]

    def p_pvector(self, p):  # pylint: disable=no-self-use
        """ pvector : pvector prob """
        p[0] = p[1] + [p[2]]

    def p_pvector_base(self, p):  # pylint: disable=no-self-use
        """ pvector : prob """
        p[0] = [p[1]]

    def p_prob(self, p):  # pylint: disable=no-self-use
        """ prob : number """
        prob = p[1]
        if not 0 <= prob <= 1:
            raise SemanticError(f'Probability value ({prob}) out of bounds.')
        p[0] = prob

    ### NUMBERS

    def p_nmatrix(self, p):  # pylint: disable=no-self-use
        """ nmatrix : nmatrix NL nvector """
        p[0] = p[1] + [p[3]]

    def p_nmatrix_base(self, p):  # pylint: disable=no-self-use
        """ nmatrix : nvector """
        p[0] = [p[1]]

    def p_nvector(self, p):  # pylint: disable=no-self-use
        """ nvector : nvector number """
        p[0] = p[1] + [p[2]]

    def p_nvector_base(self, p):  # pylint: disable=no-self-use
        """ nvector : number """
        p[0] = [p[1]]

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
