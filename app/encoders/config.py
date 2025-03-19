DEFAULT_ENCODER_CONFIG = {
    "code": {
        "model_name": "microsoft/codebert-base",
        "trust_remote_code": True,
        "device": "cpu",
        "kwargs": {},
    },
    "text": {
        "model_name": "all-MiniLM-L6-v2",
        "trust_remote_code": False,
        "device": "cpu",
        "kwargs": {},
    },
}


def get_encoder_config(encoder_type):

    """
    Get configuration for a specific encoder type.
    :param str encoder_type: Type of encoder ('code' or 'text')
    :return: dict: Configuration settings for the specified encoder
    :raises: ValueError: If the encoder_type is not supported
    """
    if encoder_type in DEFAULT_ENCODER_CONFIG:
        return DEFAULT_ENCODER_CONFIG[encoder_type]
    raise ValueError(f"Unsupported encoder type: {encoder_type}")
