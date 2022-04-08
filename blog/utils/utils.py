# Generic utility functions


def get_key(dict, val):
    """
    Return the key of a dictionary given a value

    Args:
        dict (dict): Dictionary to search.
        val ():Value to search for.

    Returns:
       Value of the key.
    """
    for key, value in dict.items():
        if val == value:
            return key
    return "Key doesn't exist"
