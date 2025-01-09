def inputs_validator(inputs):
    if not isinstance(inputs, dict) :
        raise TypeError("query must be a dict")
    missing_keys = list(set(['inputs', 'question']) - set(inputs.keys()))
    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys: {', '.join(missing_keys)}")
    return True