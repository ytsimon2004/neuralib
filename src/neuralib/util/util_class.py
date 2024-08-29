__all__ = ['check_attrs_in_clazz']


def check_attrs_in_clazz(cls: type, attr: str) -> bool:
    """
    Check if attr is in the class

    :param cls: class name
    :param attr: attribute name
    :return:
    """
    if attr in getattr(cls, '__annotations__', {}):
        return True

    for c in cls.mro()[1:]:  # Skip the first class as it's already checked
        if attr in getattr(c, '__annotations__', {}):
            return True

    return False
