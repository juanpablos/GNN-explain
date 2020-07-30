from .graphs.foc import *


formulas = {
    "5caab97089":
        AND(
            OR(
                Property("BLACK"),
                Property("GREEN")
            ),
            Exist(
                AND(
                    Role("EDGE"),
                    Property("BLUE")
                ),
                3,
                5
            )
        ),
    "7e24cdcffb": Property("RED"),
    "74a0324f6e": Property("GREEN"),
    "45207fda29": OR(
            AND(
                Property("RED"),
                Exist(
                    AND(
                        Role("EDGE"),
                        Property("RED")
                    ),
                    4
                )
            ),
            AND(
                Property("BLUE"),
                Exist(
                    AND(
                        Role("EDGE"),
                        Property("BLUE")
                    ),
                    4
                )
            ),
            AND(
                Property("GREEN"),
                Exist(
                    AND(
                        Role("EDGE"),
                        Property("GREEN")
                    ),
                    4
                )
            ),
            AND(
                Property("BLACK"),
                Exist(
                    AND(
                        Role("EDGE"),
                        Property("BLACK")
                    ),
                    4
                )
            )
    ),
    "a085814a6b": AND(
        Property("BLUE"),
        Exist(
            AND(
                Role("EDGE"),
                Property("GREEN")
            ),
            2
        )
            ),
    "b18de7fd2c": AND(
            Property("GREEN"),
            OR(
                Exist(
                    AND(
                        Role("EDGE"),
                        Property("BLUE")
                    ),
                    2,
                    4
                ),
                Exist(
                    AND(
                        Role("EDGE"),
                        Property("RED")
                    ),
                    4,
                    6
                )
            )
    ),
    "bfa11bd667": AND(
            Property("RED"),
            Exist(
                AND(
                    Role("EDGE"),
                    OR(
                        Property("BLACK"),
                        Property("BLUE"),
                    )
                ),
                2,
                4
            )
    ),
    "c716a094ab": OR(Property("BLUE"), Property("GREEN"))}
