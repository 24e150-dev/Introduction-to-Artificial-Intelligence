import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    joint_prob = 1.0

    for person in people:
        # 1. Determine the number of genes and trait status for this person
        genes = 2 if person in two_genes else 1 if person in one_gene else 0
        has_trait = person in have_trait

        # 2. Calculate the probability of them having this number of genes
        if people[person]["mother"] is None and people[person]["father"] is None:
            # No parents, use unconditional probability
            gene_prob = PROBS["gene"][genes]
        else:
            # Has parents, calculate based on inheritance and mutation
            mother = people[person]["mother"]
            father = people[person]["father"]

            # Helper function to get the probability of a parent passing the gene
            def passing_prob(parent_name):
                if parent_name in two_genes:
                    return 1 - PROBS["mutation"]
                elif parent_name in one_gene:
                    return 0.5
                else:
                    return PROBS["mutation"]

            p_mother_passes = passing_prob(mother)
            p_father_passes = passing_prob(father)

            if genes == 2:
                # Must get from both parents
                gene_prob = p_mother_passes * p_father_passes
            elif genes == 1:
                # From mother AND not father, OR from father AND not mother
                gene_prob = (p_mother_passes * (1 - p_father_passes)) + ((1 - p_mother_passes) * p_father_passes)
            else:
                # Must get from neither parent
                gene_prob = (1 - p_mother_passes) * (1 - p_father_passes)

        # 3. Calculate the probability of them having the trait given their genes
        trait_prob = PROBS["trait"][genes][has_trait]

        # 4. Multiply into the joint probability
        joint_prob *= (gene_prob * trait_prob)

    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    for person in probabilities:
        # Determine current state in this specific joint distribution
        genes = 2 if person in two_genes else 1 if person in one_gene else 0
        has_trait = person in have_trait

        # Add the joint probability 'p' to the correct running totals
        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    for person in probabilities:
        # Normalize genes
        gene_sum = sum(probabilities[person]["gene"].values())
        for gene_count in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene_count] /= gene_sum

        # Normalize traits
        trait_sum = sum(probabilities[person]["trait"].values())
        for trait_status in probabilities[person]["trait"]:
            probabilities[person]["trait"][trait_status] /= trait_sum


if __name__ == "__main__":
    main()
