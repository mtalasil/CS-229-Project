import numpy as np
# point convention is (z, y, x) (higher dimension first)
class Point():
    def __init__(self, data: list[int]):
        self.data = np.array(data, dtype=int)
        self._key = tuple(int(v) for v in self.data)

    def __add__(self, other):
        new_data = self.data + other.data
        return Point(new_data.tolist())
    
    def __sub__(self, other):
        new_data = self.data - other.data
        return Point(new_data.tolist())
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def dist(self, other):
        return np.linalg.norm(self.data - other.data)
    
    def __eq__(self, other):
        return self._key == other._key

    def __hash__(self):
        return hash(self._key)

    def __repr__(self):
        return f"Point{self._key}"
    
    
