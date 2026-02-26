from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

ASaidKnight = Symbol("A said I am a Knight")
ASaidKnave = Symbol("A said I am a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # A is either knight or knave, not both
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),

    # If A is a Knight, then his statement is true
    Implication(AKnight, And(AKnight, AKnave)),

    # If A is a Knave, then his statement is false
    Implication(AKnave, Not(And(AKnight, AKnave)))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # A structure
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),

    # B structure
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),

    # A's statement
    Implication(AKnight, And(AKnave, BKnave)),
    Implication(AKnave, Not(And(AKnave, BKnave)))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
same = Or(
    And(AKnight, BKnight),
    And(AKnave, BKnave)
)

different = Or(
    And(AKnight, BKnave),
    And(AKnave, BKnight)
)

knowledge2 = And(
    # Structure
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),

    # A's statement
    Implication(AKnight, same),
    Implication(AKnave, Not(same)),

    # B's statement
    Implication(BKnight, different),
    Implication(BKnave, Not(different))
)


# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # Rules for A, B, and C
    Or(AKnight, AKnave), Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave), Not(And(BKnight, BKnave)),
    Or(CKnight, CKnave), Not(And(CKnight, CKnave)),

    # A's first statement is either "I am a knight" or "I am a knave"
    # Note: A knight saying "I am a knight" is true; a knave saying "I am a knight" is false.
    # A knight saying "I am a knave" is impossible; a knave saying it is also impossible.
    # We represent the two possibilities for A's statement:
    Or(
        Biconditional(AKnight, AKnight),
        Biconditional(AKnight, AKnave)
    ),

    # B says: "A said 'I am a knave'."
    # This means B is a knight IF AND ONLY IF A's statement was "I am a knave"
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),

    # B says: "C is a knave."
    Biconditional(BKnight, CKnave),

    # C says: "A is a knight."
    Biconditional(CKnight, AKnight)
)

def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
