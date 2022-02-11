## psi=3: A, B, C
# layer 0: [ A, B, C ] -> psi=3
# layer 1: [ AB, AC, BA, BC, CA, CB ] -> psi * (psi-1)=6
# layer 2: [ ABC, ACB, BAC, BCA, CAB, CBA ] -> psi * (psi-1) * (psi-2)=6

## psi=3->4: A, B, C, D
# layer 0: [ A, B, C, D ]  = [ A, B, C ] + [ D ]
# layer 1: [ AB, AC, AD, BA, BC, BD, CA, CB, CD, DA, DB, DC] = [ AB, AC, BA, BC, CA, CB ] + [AD, BD, CD, DA, DB,]