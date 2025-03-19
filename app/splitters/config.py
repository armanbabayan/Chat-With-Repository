DEFAULT_CONFIG = {
    "python": {
        "chunk_size": 2000,
        "chunk_overlap": 200,
    },
    "markdown": {
        "chunk_size": 2000,
        "chunk_overlap": 0,
    },
    "default": {
        "chunk_size": 1500,
        "chunk_overlap": 100,
    },
}


def get_config(language: str = None) -> dict:
    """
    Get configuration for a specific language or default config.

    :param (str, optional) language:  Language identifier (e.g., 'python', 'markdown'). Defaults to None.
    :return: dict: Configuration settings for the specified language or default settings.

    """

    if language and language.lower() in DEFAULT_CONFIG:
        return DEFAULT_CONFIG[language.lower()]
    return DEFAULT_CONFIG["default"]
