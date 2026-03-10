import pickle


def save_pickle(data, path):
    """
    Serializes ``data`` to a pickle file at ``path``.

    :param data: The object to serialize.
    :param path: File path to write to.
    :type path: str
    """
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(path):
    """
    Deserializes and returns the object stored in the pickle file at ``path``.

    :param path: File path to read from.
    :type path: str
    :returns: The deserialized object.
    """
    with open(path, "rb") as file:
        return pickle.load(file)
