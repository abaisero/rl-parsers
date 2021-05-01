# rl-parsers

This repository contains parsers for file formats related to reinforcement
learning.  In each case, the contents of the parsed file is returned as a
`dataclass` instance containing the fields specified by the respective file
format.

## POMDP:  Partially Observable Markov Decision Process

POMDP file format (see the [POMDP File Format Page][pomdp-format]), extended by
the addition of a `reset` keyword and an `OO` rule:
* The `reset` keyword may be used both to indicate either the end of an episode
  (in episodic tasks), or the reinitialization of the state according to the
  initial state distribution (in continuing tasks).
* The `OO` rule is like the `O` rule, except that observation probabilities may
  depend on the previous state as well as the next state.

## DPOMDP:  Decentralized Partially Observable Markov Decision Process

DPOMDP file format, extended by the addition of a `reset` keyword:
* The `reset` keyword may be used both to indicate either the end of an episode
  (in episodic tasks), or the reinitialization of the state according to the
  initial state distribution (in continuing tasks).

## MDP:  Markov Decision Process

MDP file format (can be seen as a sub-set of the POMDP format), extended by the
addition of a `reset` keyword:
* The `reset` keyword may be used both to indicate either the end of an episode
  (in episodic tasks), or the reinitialization of the state according to the
  initial state distribution (in continuing tasks).

## FSC:  Finite State Controller

A FSC is a graph-based policy used for partially observable environments.  This
is my own custom file format.

[pomdp-format]: http://pomdp.org/code/pomdp-file-spec.html
