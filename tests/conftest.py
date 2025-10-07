import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT_DIR)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

try:  # pragma: no cover - executed when pandas is available
    import pandas  # noqa: F401  # pragma: no cover
except ModuleNotFoundError:  # pragma: no cover - lightweight stub for offline envs
    stub = types.ModuleType("pandas")

    class DataFrame:  # pragma: no cover - behaviour verified via tests
        def __init__(self, data):
            self._columns = list(data.keys())
            values = [list(column_values) for column_values in data.values()]
            if values:
                length = {len(column) for column in values}
                if len(length) != 1:
                    raise ValueError("All columns must share the same length")
                row_count = length.pop()
                self._rows = [
                    {column: values[idx][row] for idx, column in enumerate(self._columns)}
                    for row in range(row_count)
                ]
            else:
                self._rows = []

        @property
        def columns(self):
            return list(self._columns)

        @property
        def empty(self):
            return len(self._rows) == 0

        def copy(self):
            data = {column: [row[column] for row in self._rows] for column in self._columns}
            return DataFrame(data)

        def rename(self, *, columns):
            renamed_data = {}
            for column in self._columns:
                new_name = columns.get(column, column)
                renamed_data[new_name] = [row[column] for row in self._rows]
            return DataFrame(renamed_data)

        def equals(self, other):
            if not isinstance(other, DataFrame):
                return False
            return self._columns == other._columns and self._rows == other._rows

        class _LocAccessor:
            def __init__(self, frame):
                self._frame = frame

            def __getitem__(self, key):
                row_idx, column = key
                return self._frame._rows[row_idx][column]

        @property
        def loc(self):
            return DataFrame._LocAccessor(self)

    stub.DataFrame = DataFrame
    sys.modules["pandas"] = stub
