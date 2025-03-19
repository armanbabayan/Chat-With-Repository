class QueryClassifier:
    """Class for classifying user queries to determine retrieval weights."""

    def __init__(self, llm):

        self.llm = llm

    def classify(self, query, categories=None):
        """
        Classify a query into predefined categories.
        :param str query: User query to classify
        :param list categories: List of categories to choose from
        :return list: Weights for each category
        """

        if not categories:
            categories = ["documentation", "code"]

        category_str = "', '".join(categories)
        prompt = f"Classify if this query is about {category_str}: '{query}'. Answer with one of: '{category_str}', or 'both'."

        result = self.llm.generate([prompt]).generations[0][0].text.strip().lower()

        # Default weights (equal distribution)
        weights = [0.5, 0.5]

        # Adjust weights based on classification
        if result == "both":
            return weights  # Equal weights

        for i, category in enumerate(categories):
            if result == category:
                # Assign higher weight to the matched category
                weights = [0.3] * len(categories)
                weights[i] = 0.7
                break

        return weights

    def get_retriever_weights(self, query, retrievers=None):
        """
        Get weights for retriever ensemble based on query classification.
        :param str query: User query to classify
        :param list retrievers: List of retriever names
        :return list: Weights for each retriever
        """

        if not retrievers:
            retrievers = ["documentation", "code"]

        return self.classify(query, retrievers)
