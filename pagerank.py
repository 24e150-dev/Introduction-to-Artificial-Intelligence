import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    """
    distribution = {}
    links = corpus[page]
    pages_count = len(corpus)

    # If page has no links, treat it as having links to all pages
    if not links:
        prob = 1 / pages_count
        return {p: prob for p in corpus}

    # Probability of picking a page at random (1 - d)
    random_prob = (1 - damping_factor) / pages_count

    # Probability of following a specific link (d / number_of_links)
    link_prob = damping_factor / len(links)

    for p in corpus:
        distribution[p] = random_prob
        if p in links:
            distribution[p] += link_prob

    return distribution

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling n pages
    starting with a page at random.
    """
    counts = {page: 0 for page in corpus}
    
    # First sample is chosen at random
    current_page = random.choice(list(corpus.keys()))
    counts[current_page] += 1

    # Generate remaining n-1 samples
    for _ in range(n - 1):
        model = transition_model(corpus, current_page, damping_factor)
        pages = list(model.keys())
        weights = list(model.values())
        
        # Select next page based on transition model probabilities
        current_page = random.choices(pages, weights=weights, k=1)[0]
        counts[current_page] += 1

    # Convert counts to proportions
    return {page: count / n for page, count in counts.items()}

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    values until convergence.
    """
    n = len(corpus)
    # Start with equal rank 1/N
    ranks = {page: 1 / n for page in corpus}
    
    # Pre-process: pages with no links link to everything
    new_corpus = {page: (links if links else set(corpus.keys())) 
                  for page, links in corpus.items()}

    while True:
        new_ranks = {}
        for p in corpus:
            # First part of formula: (1 - d) / N
            rank_sum = (1 - damping_factor) / n
            
            # Second part: d * sum(PR(i) / NumLinks(i))
            link_contribution = 0
            for i in corpus:
                if p in new_corpus[i]:
                    link_contribution += ranks[i] / len(new_corpus[i])
            
            new_ranks[p] = rank_sum + (damping_factor * link_contribution)

        # Check for convergence (difference < 0.001)
        if all(abs(new_ranks[p] - ranks[p]) < 0.001 for p in ranks):
            return new_ranks
        
        ranks = new_ranks

# (Rest of the CS50 provided crawl function would go below)
def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a set of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


if __name__ == "__main__":
    main()
