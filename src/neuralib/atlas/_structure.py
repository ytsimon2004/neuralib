from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypedDict

import allensdk.core.structure_tree
from allensdk.core.reference_space_cache import ReferenceSpaceCache

from neuralib.util.io import ALLEN_SDK_DIRECTORY
from neuralib.util.util_type import PathLike

__all__ = ['AllenReferenceWrapper']


class StructureTreeDict(TypedDict):
    """allen structure tree dict"""
    acronym: str
    graph_id: int
    graph_order: int
    id: int
    name: str
    structure_id_path: list[int]
    structure_set_ids: list[int]
    rgb_triplet: list[int]


class AllenReferenceWrapper:
    """Class for load the data in allen"""

    REFERENCE_SPACE_KEY: ClassVar[str] = 'allensdk'
    STRUCTURE_GRAPH_ID: ClassVar[int] = 1

    def __init__(self,
                 resolution: int = 10,
                 output: PathLike | None = None):
        """
        :param resolution: resolution in um
        :param output: output directory for downloading cache. By default, save in ``CCF_CACHE_DIRECTORY``
        """

        if output is None:
            output = ALLEN_SDK_DIRECTORY

        if not output.exists():
            output.mkdir(parents=True, exist_ok=True)

        self.source_root = Path(output)
        self.reference = ReferenceSpaceCache(resolution,
                                             self.REFERENCE_SPACE_KEY,
                                             manifest=self.source_root / 'manifest.json')

    def structure_tree(self) -> allensdk.core.structure_tree.StructureTree:
        return self.reference.get_structure_tree(structure_graph_id=self.STRUCTURE_GRAPH_ID)

    def get_structures_by_name(self, name: list[str]) -> list[StructureTreeDict]:
        return self.structure_tree().get_structures_by_name(name)
