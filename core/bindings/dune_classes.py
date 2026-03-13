import numpy as np

from dune.istl._istl import BlockVector as DuneVector

from pymor.models.basic import Model
from pymor.operators.interface import Operator
from pymor.parameters.base import Parameters
from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace


class DuneMatrix:
    def __init__(self, matrix):
        self.impl_ = matrix

    def __getattr__(self, name):
        return getattr(self.impl_, name)

    @property
    def shape(self):
        return self.impl_.shape

    def mv(self, x, y):
        self.impl_.mv(x, y)

    def toarray(self, dtype=np.float64):
        dense = np.zeros(self.shape, dtype=dtype)
        for i, row in self.impl_.enumerate:
            for j, block in row.enumerate:
                if block.rows != 1 or block.cols != 1:
                    raise NotImplementedError("Only scalar block matrices are supported.")
                dense[i, j] = block[0][0]
        return dense


class WrappedDuneVector(CopyOnWriteVector):
    def __init__(self, vector):
        assert isinstance(vector, DuneVector)
        self.impl_ = vector

    @classmethod
    def from_instance(cls, instance):
        return cls(instance.impl_)

    def to_numpy(self, ensure_copy=False):
        arr = np.asarray(self.impl_, dtype=np.float64).reshape(-1)
        return arr.copy() if ensure_copy else arr

    def _copy_data(self):
        self.impl_ = self.impl_.copy()

    def _scal(self, alpha):
        self.impl_ *= alpha

    def _axpy(self, alpha, x):
        self.impl_ += x.impl_ * alpha

    def inner(self, other):
        return self.impl_ * other.impl_

    def norm(self):
        return float(self.impl_.two_norm)

    def norm2(self):
        n = float(self.impl_.two_norm)
        return n * n

    def sup_norm(self):
        return float(self.impl_.infinity_norm)

    def dofs(self, dof_indices):
        return self.to_numpy()[dof_indices]

    def amax(self):
        data = np.abs(self.to_numpy())
        idx = int(np.argmax(data))
        return idx, float(data[idx])

    def __len__(self):
        return len(self.impl_)


class DuneVectorSpace(ListVectorSpace):
    vector_type = WrappedDuneVector

    def __init__(self, dim, id="STATE"):
        self.dim = int(dim)
        self.id = id

    def __eq__(self, other):
        return type(other) is DuneVectorSpace and self.dim == other.dim and self.id == other.id

    def __hash__(self):
        return hash((self.dim, self.id))

    @classmethod
    def space_from_vector_obj(cls, vec, id):
        return cls(len(vec), id=id)

    @classmethod
    def space_from_dim(cls, dim, id):
        return cls(dim, id=id)

    def zero_vector(self):
        vec = DuneVector(self.dim)
        vec *= 0.0
        return WrappedDuneVector(vec)

    def make_vector(self, obj):
        if isinstance(obj, WrappedDuneVector):
            return obj
        return WrappedDuneVector(obj)

    def vector_from_numpy(self, data, ensure_copy=False):
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
        assert arr.size == self.dim
        vec = DuneVector(self.dim)
        vec[:] = arr.copy() if ensure_copy else arr
        return WrappedDuneVector(vec)


class DuneL2Product(Operator):
    linear = True

    def __init__(self, solver):
        self.solver = solver
        self.matrix = DuneMatrix(solver.getL2MassMatrix())
        self.source = DuneVectorSpace(solver.dim_source, id="STATE")
        self.range = DuneVectorSpace(solver.dim_range, id="STATE")

    def apply(self, U, mu=None):
        assert U in self.source
        V = self.range.zeros(len(U))
        for u, v in zip(U.vectors, V.vectors):
            self.matrix.mv(u.impl_, v.impl_)
        return V


class DuneDarcyFlowModel(Model):
    def __init__(
        self,
        solver,
        config,
        products=None,
        parameter_indices=None,
        error_estimator=None,
        name="dune-darcyflow model",
    ):
        self.solver = solver
        self.config = config
        self.solution_space = DuneVectorSpace(solver.dim_range, id="STATE")
        self._base_parameter = np.array(
            [
                float(config.get("problem.parametric.coatingHeight", 0.0)),
                float(config.get("problem.parametric.inflowAngle", 0.0)),
                float(config.get("problem.parametric.minPermeability", 0.0)),
                float(config.get("problem.parametric.coatingPermeability", 0.0)),
            ],
            dtype=np.float64,
        )
        self.parameter_indices = parameter_indices or {"mu": (0, 1), "nu": (2, 3)}
        super().__init__(products=products, error_estimator=error_estimator, name=name)
        self.parameters_own = Parameters({k: len(v) for k, v in self.parameter_indices.items()})

    def _solver_parameter(self, mu):
        p = self._base_parameter.copy()
        if mu is None:
            return p
        for name, indices in self.parameter_indices.items():
            if name not in mu:
                continue
            vals = np.asarray(mu[name], dtype=np.float64).reshape(-1)
            if vals.size != len(indices):
                raise ValueError(f"Expected {len(indices)} entries for '{name}', got {vals.size}")
            p[np.asarray(indices, dtype=int)] = vals
        return p

    def _compute_solution(self, mu=None, **kwargs):
        trajectory = self.solver.solve(self._solver_parameter(mu).tolist())
        return self.solution_space.make_array(trajectory)

    def visualize(self, U, **kwargs):
        filename = kwargs.get("filename", "solution")
        if hasattr(U, "vectors"):
            self.solver.visualize([v.impl_ for v in U.vectors], filename)
        else:
            self.solver.visualize(U, filename)
