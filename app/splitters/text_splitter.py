from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language
from .config import get_config


class TextSplitterFactory:

    """
    Factory class for creating TextSplitters for different languages.
    """

    @classmethod
    def create_splitter(cls, language_name, custom_config=None):
        """
        Create a text splitter for the specified language.

        :param str language_name: The language for which to create a splitter (e.g., 'python', 'markdown').
        :param dict custom_config: Custom configuration to override defaults. Defaults to None.
        :return: RecursiveCharacterTextSplitter: A configured text splitter for the language.
        :raises ValueError: If language is not supported.
        """

        try:
            language = getattr(Language, language_name.upper())
        except (AttributeError, TypeError):
            raise ValueError(f"Unsupported language: {language_name}")

        config = get_config(language_name)

        if custom_config:
            config.update(custom_config)

        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )


def get_python_splitter(custom_config=None):
    """
    Get a Python text splitter with default or custom configuration.

    :param dict custom_config: Custom configuration to override defaults.
    :return: RecursiveCharacterTextSplitter: A configured text splitter for the language.
    """
    return TextSplitterFactory.create_splitter("python", custom_config)


def get_markdown_splitter(custom_config=None):
    """
    Get a Markdown text splitter with default or custom configuration.
    :param dict custom_config: Custom configuration to override defaults.
    :return: RecursiveCharacterTextSplitter: A configured text splitter for the language.
    """
    return TextSplitterFactory.create_splitter("markdown", custom_config)
