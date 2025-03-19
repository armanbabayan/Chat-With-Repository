class Config:
    VECTOR_PARAMS = {
        "vector_size": 384,
        "distance_metric": "COSINE",  # Options: COSINE, EUCLID, DOT
        "batch_size": 100,
    }

    @classmethod
    def get(cls, key, default=None):

        """Get a configuration value by key with optional default"""
        parts = key.split(".")

        current = cls.__dict__
        for part in parts:
            if part.upper() in current:
                current = current[part.upper()]
            elif part in current:
                current = current[part]
            else:
                return default

        return current
