# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Any, List, Tuple, Union
import torch
from torch.nn import functional as F


class Lines:
    def __init__(self, lines: List[Union[torch.Tensor, np.ndarray]]):
        if not isinstance(lines, list):
            raise ValueError(
                "Cannot create Lines: Expect a list of list of lines per image. "
                "Got '{}' instead.".format(type(lines))
            )
        
        def _make_array(t: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
            # Use float64 for higher precision, because why not?
            # Always put lines on CPU (self.to is a no-op) since they
            # are supposed to be small tensors.
            # May need to change this assumption if GPU placement becomes useful
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()
            return np.asarray(t).astype("float64")
        
        def process_lines(
            lines_per_instance: List[Union[torch.Tensor, np.ndarray]]
        ) -> List[np.ndarray]:
            # transform each line to a numpy array
            if len(lines_per_instance) != 1:
                raise ValueError(f"Cannot create a line from {len(lines_per_instance)} coordinates.")
            
            lines_per_instance = _make_array(lines_per_instance)

            return lines_per_instance

        self.lines: List[np.ndarray] = [
            process_lines(lines_per_instance) for lines_per_instance in lines
        ]

    def to(self, *args: Any, **kwargs: Any) -> "Lines":
        return self

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")
    
    def __len__(self) -> int:
        return len(self.lines)
    
    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Lines":
        if isinstance(item, int):
            selected_polygons = [self.lines[item]]
        elif isinstance(item, slice):
            selected_polygons = self.lines[item]
        elif isinstance(item, list):
            selected_polygons = [self.lines[i] for i in item]
        elif isinstance(item, torch.Tensor):
            # Polygons is a list, so we have to move the indices back to CPU.
            if item.dtype == torch.bool:
                assert item.dim() == 1, item.shape
                item = item.nonzero().squeeze(1).cpu().numpy().tolist()
            elif item.dtype in [torch.int32, torch.int64]:
                item = item.cpu().numpy().tolist()
            else:
                raise ValueError("Unsupported tensor dtype={} for indexing!".format(item.dtype))
            selected_polygons = [self.lines[i] for i in item]
        return Lines(selected_polygons)
    
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.lines))
        return s