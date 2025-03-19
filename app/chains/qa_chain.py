from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)


class QAChainBuilder:
    """Builder class for creating QA chains with retrieval and reranking."""

    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = """Answer the question strictly based on the following GitHub
                                  repository context. Do not use any external knowledge or
                                  make assumptions. If the question is outside the given
                                  context or cannot be answered based on the provided information,
                                  respond with 'I don't know.'

                                {context}

                                Question: {question}
                                """

    def set_prompt_template(self, template):
        """
        Set a custom prompt template.

        :param str template: Custom prompt template
        :return QAChainBuilder: Self for method chaining:
        """

        self.prompt_template = template
        return self

    def build_with_ensemble(self, ensemble_retriever, reranker=None):
        """
        Build a QA chain with ensemble retrieval and reranking.
        :param ensemble_retriever: Ensemble retriever instance
        :param reranker: Reranker instance
        :return: Runnable chain
        """

        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        retrieval_step = RunnableParallel(
            {"context": ensemble_retriever, "question": RunnablePassthrough()}
        )

        # Add reranking if provided
        if reranker:
            retrieval_step = retrieval_step | RunnableLambda(reranker.format_for_prompt)

        # Build the full chain
        chain = retrieval_step | prompt | self.llm | StrOutputParser()

        return chain

    def build_with_dynamic_weights(self, retrievers, classifier, reranker=None):
        """
        Build a QA chain that dynamically determines retriever weights.

        :param list retrievers: List of retriever instances
        :param QueryClassifier classifier: Query classifier for determining weights
        :param reranker reranker: Reranker instance
        :return chain: Runnable chain
        """

        from langchain.retrievers import EnsembleRetriever

        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        def get_dynamic_ensemble(query):
            weights = classifier.get_retriever_weights(query)
            return EnsembleRetriever(retrievers=retrievers, weights=weights)

        # Create the retrieval step with dynamic weights
        def retrieve_with_dynamic_weights(query):
            dynamic_retriever = get_dynamic_ensemble(query)
            results = dynamic_retriever.invoke(query)
            return {"context": results, "question": query}

        retrieval_step = RunnableLambda(retrieve_with_dynamic_weights)

        if reranker:
            retrieval_step = retrieval_step | RunnableLambda(reranker.format_for_prompt)

        chain = retrieval_step | prompt | self.llm | StrOutputParser()

        return chain
