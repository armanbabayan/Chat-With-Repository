def answer_question(query, dynamic_chain=None, static_chain=None):

    """
    Answer a question using the QA system.
    :param str query:
    :param chain dynamic_chain: Chain with dynamic weighting
    :param static_chain: Chain with static weighting ([0.5, 05])
    :return str: Answer to the question
    """
    chain = dynamic_chain if dynamic_chain else static_chain
    return chain.invoke(query)
