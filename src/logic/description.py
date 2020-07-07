class DescriptionLogic:
    pass



class Concept:
    """
    RED(x)
    """
    pass


class Operator:
    pass


class Negation(Operator):
    pass


class Conj(Operator):
    pass


class Disj(Operator):
    pass


class Role:
    """
    ROLE is a binary function
    ex. child(x, y) === y is a child of x
    """
    pass


class Restriction:
    pass


class Existential(Restriction):
    """
    EXIST R.C === x such that E y (x, y) in R AND y in C
    """
    pass


class Universal(Restriction):
    """
    FORALL R.C === x such that A y (x, y) in R -> y in C
    """
    pass


class Qualified:
    """
    (>=N R.C) === exist 1..N different values that
    satisfies (x, x_i) in R AND x_i in C
    """
    pass
