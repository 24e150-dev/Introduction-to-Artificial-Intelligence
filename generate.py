import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.crossword.variables:
            # Iterate over a copy of the domain so we can safely remove elements
            for word in self.domains[var].copy():
                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]
        
        # If there is no overlap, no revision is needed
        if overlap is None:
            return False
        
        i, j = overlap
        
        for word_x in self.domains[x].copy():
            # Check if there is ANY word in y's domain that matches the overlapping character
            conflict = True
            for word_y in self.domains[y]:
                if word_x[i] == word_y[j]:
                    conflict = False
                    break
                    
            # If no word in y's domain works with word_x, remove word_x
            if conflict:
                self.domains[x].remove(word_x)
                revised = True
                
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # Initialize the queue with provided arcs or all possible arcs
        if arcs is None:
            queue = []
            for var in self.crossword.variables:
                for neighbor in self.crossword.neighbors(var):
                    queue.append((var, neighbor))
        else:
            queue = list(arcs)
            
        while queue:
            x, y = queue.pop(0)
            
            # If we made a revision to x's domain, we need to re-check its neighbors
            if self.revise(x, y):
                # If a domain is emptied, the puzzle is unsolvable
                if len(self.domains[x]) == 0:
                    return False
                    
                # Add all neighbors of x (excluding y) back to the queue
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))
                    
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Check if every variable in the crossword is present in the assignment dictionary
        return set(assignment.keys()) == self.crossword.variables

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Check 1: All values must be distinct
        if len(set(assignment.values())) != len(assignment.values()):
            return False
            
        for var, word in assignment.items():
            # Check 2: Word length must match variable length
            if var.length != len(word):
                return False
                
            # Check 3: Check for conflicts with assigned neighbors
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    if word[i] != assignment[neighbor][j]:
                        return False
                        
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for unassigned neighbors.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        def count_eliminated(value):
            eliminated = 0
            # Check all unassigned neighbors
            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    # Count how many words in the neighbor's domain conflict with this value
                    for neighbor_word in self.domains[neighbor]:
                        if value[i] != neighbor_word[j]:
                            eliminated += 1
            return eliminated

        # Return the list of values in the domain sorted by the elimination count
        return sorted(list(self.domains[var]), key=count_eliminated)

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned = set(self.crossword.variables) - set(assignment.keys())
        
        # Sort by minimum remaining values (ascending), then by degree (descending)
        sorted_unassigned = sorted(
            list(unassigned),
            key=lambda var: (len(self.domains[var]), -len(self.crossword.neighbors(var)))
        )
        
        return sorted_unassigned[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # Base case: If the assignment is complete, we're done
        if self.assignment_complete(assignment):
            return assignment
            
        # Select the next unassigned variable to explore
        var = self.select_unassigned_variable(assignment)
        
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            # Check if this new assignment is consistent with the constraints
            if self.consistent(new_assignment):
                
                # Recursive call with the new valid assignment
                result = self.backtrack(new_assignment)
                
                # If the result isn't None, we found a successful complete assignment
                if result is not None:
                    return result
                    
        # If all domain values fail, backtrack
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()