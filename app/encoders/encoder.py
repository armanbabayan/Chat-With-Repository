from sentence_transformers import SentenceTransformer
from app.encoders.config import get_encoder_config


class EncoderFactory:
    """Factory class for creating and managing text and code encoders."""

    _encoders = {}  # Class-level cache for encoders

    @classmethod
    def get_encoder(cls, encoder_type, custom_config=None):
        """
        Get an encoder for the specified type, reusing existing instances when possible.

        :param str encoder_type: type of encoder ('code' or 'text')
        :param dict custom_config: Custom configuration to override defaults
        :return: SentenceTransformer: The encoder model
        :raises: ValueError: If the encoder_type is not supported
        """

        # Create a config key that includes any custom settings
        config_key = encoder_type
        if custom_config:
            config_key = f"{encoder_type}_{hash(frozenset(custom_config.items()))}"

        # Return cached encoder if available
        if config_key in cls._encoders:
            return cls._encoders[config_key]

        # Get config and override with custom settings if provided
        config = get_encoder_config(encoder_type).copy()
        if custom_config:
            config.update(custom_config)

        # Extract parameters
        model_name = config.pop("model_name")
        kwargs = config.pop("kwargs", {})

        # Merge remaining config items into kwargs
        kwargs.update(config)

        # Create the encoder
        encoder = SentenceTransformer(model_name, **kwargs)

        # Cache and return
        cls._encoders[config_key] = encoder
        return encoder

    @classmethod
    def clear_cache(cls):

        """Clear the encoder cache to free memory."""

        cls._encoders = {}


def get_code_encoder(custom_config=None):
    """Get a code encoder with default or custom configuration."""
    return EncoderFactory.get_encoder("code", custom_config)


def get_text_encoder(custom_config=None):
    """Get a text encoder with default or custom configuration."""
    return EncoderFactory.get_encoder("text", custom_config)
