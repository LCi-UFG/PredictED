import random

from utils import decode_one_hot, one_hot
from rules import (
    ATOMIC_NUMBER,  
    ATOMIC_SUBSTITUTIONS, 
    CHARGE_SUBSTITUTIONS, 
    HYBRIDIZATION_SUBSTITUTIONS
    )

 
def atomic_rules(atom_one_hot):
    atom_num = decode_one_hot(
        atom_one_hot, 
        ATOMIC_NUMBER()
        )
    if atom_num in ATOMIC_SUBSTITUTIONS:
        sub_atom_num = random.choice(
            ATOMIC_SUBSTITUTIONS[atom_num])
        sub_atom_one_hot = one_hot(
            sub_atom_num, ATOMIC_NUMBER()
            )
        return sub_atom_one_hot
    else:
        return atom_one_hot


def charge_rules(
    ATOMIC_NUMBER, 
    original_charge_one_hot):

    original_charge = decode_one_hot(
        original_charge_one_hot, [-1, 0, 1, 2]
        )
    if ATOMIC_NUMBER in CHARGE_SUBSTITUTIONS:
        possible_charges = CHARGE_SUBSTITUTIONS[ATOMIC_NUMBER]
        possible_charges = [
            charge for charge in possible_charges 
            if charge != original_charge
            ]
        if possible_charges:
            perturbed_charge = random.choice(possible_charges)
            return perturbed_charge
        else:
            return original_charge
    else:
        return original_charge
    

def hybridization_rules(
    ATOMIC_NUMBER, 
    hybrid_one_hot):

    original_hybrid = decode_one_hot(
        hybrid_one_hot, list(range(5))
        )
    if ATOMIC_NUMBER in HYBRIDIZATION_SUBSTITUTIONS:
        possible_hybrid = HYBRIDIZATION_SUBSTITUTIONS[
            ATOMIC_NUMBER]['types']
        possible_hybrid = [
            hybrid for hybrid in possible_hybrid 
            if hybrid != original_hybrid
            ]
        if possible_hybrid:
            perturbed_hybrid = random.choice(possible_hybrid)
            return perturbed_hybrid
        else:
            return original_hybrid
    else:
        return original_hybrid