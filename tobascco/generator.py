# -*- coding: utf-8 -*-
import itertools

from .sbu import SBU_list


class Generate(object):
    """Takes as input a sequence of sbus, and returns
    build orders to make structures.

    """

    def __init__(self, options, sbu_list):
        self.options = options
        self.sbus = SBU_list(sbu_list)

    def generate_sbu_combinations(self, incidence=None, N=None):
        if N is None:
            N = (
                self.options.metal_sbu_per_structure
                + self.options.organic_sbu_per_structure
            )
        for i in itertools.combinations_with_replacement(self.sbus.list, N):
            if self._valid_sbu_combination(incidence, i):
                yield tuple(i)

    def combinations_from_options(self):
        """Just return the tuples in turn."""
        combs = []
        Nmetals = self.options.metal_sbu_per_structure
        for combo in self.options.sbu_combinations:
            # first sbus have to be metals.
            met = []
            for i in range(Nmetals):
                met.append(self.sbus.get(combo[i], _METAL=True))
            combs.append(tuple(met + [self.sbus.get(i) for i in combo[Nmetals:]]))
        return combs

    def _valid_sbu_combination(self, incidence, sbu_set):
        """Currently only checks if there is the correct number of metal
        SBUs in the combination."""
        # check if all the special bonds can be satisfied
        constraints = []
        specials = []
        none_const = {}
        none_spec = {}
        children = []
        for kk in sbu_set:
            if kk.children:
                for j in kk.children:
                    children.append(j)

        sbu_set = tuple(list(sbu_set) + children)
        for sbu in sbu_set:
            for cp in sbu.connect_points:
                if cp.special:
                    specials.append(cp.special)
                if cp.constraint:
                    constraints.append(cp.constraint)
                if cp.constraint is None:
                    none_const[(sbu.name, sbu.is_metal)] = cp.identifier
                if cp.special is None:
                    none_spec[(sbu.name, sbu.is_metal)] = cp.identifier

        condition1 = set(specials) == set(constraints)
        condition2 = len([i for i in sbu_set if i.is_metal]) == \
                self.options.metal_sbu_per_structure
        condition3 = False
        for sbu, met in none_spec.keys():
            for sbu2, met2 in none_const.keys():
                if met != met2:
                    condition3 = True
                    break

        if incidence is None:
            return ( 
                (len([i for i in sbu_set if i.is_metal]) == self.options.metal_sbu_per_structure)
             and condition1 and condition2 and condition3
             )
        else:
            if set(sorted([i.degree for i in sbu_set])) == set(sorted(incidence)):
                return (
                    (len([i for i in sbu_set if i.is_metal]) == self.options.metal_sbu_per_structure)
                    and condition1 and condition2 and condition3
                )
            else:
                return False

    def linear_in_combo(self, combo):
        for i in combo:
            for j in self.sbus.list:
                if j == i:
                    if j.linear or j.two_connected:
                        return True
        return False

    def yield_linear_org_sbu(self, combo):
        for i in self.sbus.list:
            if (i.linear or i.two_connected) and not i.is_metal:
                ret = list(combo) + [i]
                yield tuple(ret)

    def _valid_bond_pair(self, set):
        """Determine if the two SBUs can be bonded.  Currently set to
        flag true if the two sbus contain matching bond flags, otherwise
        if they are a (metal|organic) pair
        """
        (sbu1, cp1), (sbu2, cp2) = set
        if all(
            [
                i is None
                for i in [cp1.special, cp2.special, cp1.constraint, cp2.constraint]
            ]
        ):
            return sbu1.is_metal != sbu2.is_metal

        return (cp1.special == cp2.constraint) and (cp2.special == cp1.constraint)

    def generate_build_directives(self, sbu, sbus):
        """Requires maximum length of sbu insertions."""

        # insert metal first
        if sbu is None:
            # chose a metal (at random, if more than one)
            #NB the _yield_bonding_sbus recursion takes too long for the met7 Zr SBU. This is likely
            # due to the 12 connection sites it possesses.
            sbu = choice([x for x in sbus if x.is_metal])
        # expand the current SBU's bonds and establish possible SBU bondings
        # generate exaustive list of sbu combinations.
        for k in self._yield_bonding_sbus(sbu, set(sbus), 
                p=[0 for i in range(self.options.structure_sbu_length)]):
            yield [sbu] + k
        
    def flatten(self, s):
        """Returns a flattened list"""
        if s == []:
            return s
        if isinstance(s[0], list):
            return self.flatten(s[0]) + self.flatten(s[1:])
        return s[:1] + self.flatten(s[1:])

    def roundrobin(self, *iterables):
        pending = len(iterables)
        nexts = itertools.cycle(iter(it).__next__ for it in iterables)
        while pending:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                pending -= 1
                nexts = itertools.cycle(itertools.islice(nexts, pending))

    def all_bonds(self, it1, it2):
        for i, j in itertools.izip(it1, it2):
            yield i
            yield j

    def _gen_bonding_sbus(self, sbu, sbus, index=0):
        """Returns an iterator which runs over tuples of bonds
        with other sbus."""
        # an iterator that iterates sbu's first, then sbus' connect_points
        ncps = len(sbu.connect_points)
        sbu_repr = list(itertools.product([sbu], sbu.connect_points))

        # This becomes combinatorially intractable for met7 with 12 connection points
        bond_iter = list(self.roundrobin(*[itertools.product([s], s.connect_points) for s in sbus ]))
                                   # if s.name != sbu.name]))
        # don't like how this iterates, but will do for now.

    @property
    def linear_sbus_exist(self):
        try:
            return self._linear_exist
        except AttributeError:
            self._linear_exist = False
            for i in self.sbus.list:
                # not necessarily linear, but 2-c SBUs are OK for this function
                if i.linear or i.two_connected:
                    self._linear_exist = True
                    break
            return self._linear_exist

