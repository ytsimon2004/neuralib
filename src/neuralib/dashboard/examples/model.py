def list_animals() -> list[str]:
    return ['A', 'B', 'C']


def list_date(animal: str) -> list[str]:
    return {
        'A': ['1', '2', '3'],
        'B': ['11', '12', '13'],
        'C': ['21', '22', '23'],
    }[animal]
