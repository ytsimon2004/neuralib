from typing import TypeVar

__all__ = ['check_attrs_in_clazz']

C = TypeVar('C')


def check_attrs_in_clazz(clazz: type[C], attr: str) -> bool:
    """
    Check if attr is in the class

    :param clazz: class name
    :param attr: attribute name
    :return:
    """
    if attr in getattr(clazz, '__annotations__', {}):
        return True

    for cls in clazz.mro()[1:]:  # Skip the first class as it's already checked
        if attr in getattr(cls, '__annotations__', {}):
            return True

    return False
