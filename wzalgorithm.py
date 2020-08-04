# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _wzalgorithm
else:
    import _wzalgorithm

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _wzalgorithm.delete_SwigPyIterator

    def value(self):
        return _wzalgorithm.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _wzalgorithm.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _wzalgorithm.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _wzalgorithm.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _wzalgorithm.SwigPyIterator_equal(self, x)

    def copy(self):
        return _wzalgorithm.SwigPyIterator_copy(self)

    def next(self):
        return _wzalgorithm.SwigPyIterator_next(self)

    def __next__(self):
        return _wzalgorithm.SwigPyIterator___next__(self)

    def previous(self):
        return _wzalgorithm.SwigPyIterator_previous(self)

    def advance(self, n):
        return _wzalgorithm.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _wzalgorithm.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _wzalgorithm.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _wzalgorithm.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _wzalgorithm.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _wzalgorithm.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _wzalgorithm.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _wzalgorithm:
_wzalgorithm.SwigPyIterator_swigregister(SwigPyIterator)

class VecDouble(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _wzalgorithm.VecDouble_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _wzalgorithm.VecDouble___nonzero__(self)

    def __bool__(self):
        return _wzalgorithm.VecDouble___bool__(self)

    def __len__(self):
        return _wzalgorithm.VecDouble___len__(self)

    def __getslice__(self, i, j):
        return _wzalgorithm.VecDouble___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _wzalgorithm.VecDouble___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _wzalgorithm.VecDouble___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _wzalgorithm.VecDouble___delitem__(self, *args)

    def __getitem__(self, *args):
        return _wzalgorithm.VecDouble___getitem__(self, *args)

    def __setitem__(self, *args):
        return _wzalgorithm.VecDouble___setitem__(self, *args)

    def pop(self):
        return _wzalgorithm.VecDouble_pop(self)

    def append(self, x):
        return _wzalgorithm.VecDouble_append(self, x)

    def empty(self):
        return _wzalgorithm.VecDouble_empty(self)

    def size(self):
        return _wzalgorithm.VecDouble_size(self)

    def swap(self, v):
        return _wzalgorithm.VecDouble_swap(self, v)

    def begin(self):
        return _wzalgorithm.VecDouble_begin(self)

    def end(self):
        return _wzalgorithm.VecDouble_end(self)

    def rbegin(self):
        return _wzalgorithm.VecDouble_rbegin(self)

    def rend(self):
        return _wzalgorithm.VecDouble_rend(self)

    def clear(self):
        return _wzalgorithm.VecDouble_clear(self)

    def get_allocator(self):
        return _wzalgorithm.VecDouble_get_allocator(self)

    def pop_back(self):
        return _wzalgorithm.VecDouble_pop_back(self)

    def erase(self, *args):
        return _wzalgorithm.VecDouble_erase(self, *args)

    def __init__(self, *args):
        _wzalgorithm.VecDouble_swiginit(self, _wzalgorithm.new_VecDouble(*args))

    def push_back(self, x):
        return _wzalgorithm.VecDouble_push_back(self, x)

    def front(self):
        return _wzalgorithm.VecDouble_front(self)

    def back(self):
        return _wzalgorithm.VecDouble_back(self)

    def assign(self, n, x):
        return _wzalgorithm.VecDouble_assign(self, n, x)

    def resize(self, *args):
        return _wzalgorithm.VecDouble_resize(self, *args)

    def insert(self, *args):
        return _wzalgorithm.VecDouble_insert(self, *args)

    def reserve(self, n):
        return _wzalgorithm.VecDouble_reserve(self, n)

    def capacity(self):
        return _wzalgorithm.VecDouble_capacity(self)
    __swig_destroy__ = _wzalgorithm.delete_VecDouble

# Register VecDouble in _wzalgorithm:
_wzalgorithm.VecDouble_swigregister(VecDouble)

WINZENT_ALGORITHM_VERSION = _wzalgorithm.WINZENT_ALGORITHM_VERSION
class Result(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    error = property(_wzalgorithm.Result_error_get, _wzalgorithm.Result_error_set)
    parameters = property(_wzalgorithm.Result_parameters_get, _wzalgorithm.Result_parameters_set)

    def __init__(self):
        _wzalgorithm.Result_swiginit(self, _wzalgorithm.new_Result())
    __swig_destroy__ = _wzalgorithm.delete_Result

# Register Result in _wzalgorithm:
_wzalgorithm.Result_swigregister(Result)

class REvol(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    @staticmethod
    def pt1(y, u, t):
        return _wzalgorithm.REvol_pt1(y, u, t)

    @staticmethod
    def agePopulation(population):
        return _wzalgorithm.REvol_agePopulation(population)

    def __init__(self):
        _wzalgorithm.REvol_swiginit(self, _wzalgorithm.new_REvol())

    def frandom(self):
        return _wzalgorithm.REvol_frandom(self)

    def maxEpochs(self, *args):
        return _wzalgorithm.REvol_maxEpochs(self, *args)

    def maxNoSuccessEpochs(self, *args):
        return _wzalgorithm.REvol_maxNoSuccessEpochs(self, *args)

    def populationSize(self, *args):
        return _wzalgorithm.REvol_populationSize(self, *args)

    def eliteSize(self, *args):
        return _wzalgorithm.REvol_eliteSize(self, *args)

    def gradientWeight(self, *args):
        return _wzalgorithm.REvol_gradientWeight(self, *args)

    def successWeight(self, *args):
        return _wzalgorithm.REvol_successWeight(self, *args)

    def targetSuccess(self, *args):
        return _wzalgorithm.REvol_targetSuccess(self, *args)

    def eamin(self, *args):
        return _wzalgorithm.REvol_eamin(self, *args)

    def ebmin(self, *args):
        return _wzalgorithm.REvol_ebmin(self, *args)

    def ebmax(self, *args):
        return _wzalgorithm.REvol_ebmax(self, *args)

    def startTTL(self, *args):
        return _wzalgorithm.REvol_startTTL(self, *args)

    def measurementEpochs(self, *args):
        return _wzalgorithm.REvol_measurementEpochs(self, *args)

    def generateOrigin(self, dimensions):
        return _wzalgorithm.REvol_generateOrigin(self, dimensions)

    def generateInitialPopulation(self, origin):
        return _wzalgorithm.REvol_generateInitialPopulation(self, origin)

    def modifyIndividual(self, individual, population, currentSuccess):
        return _wzalgorithm.REvol_modifyIndividual(self, individual, population, currentSuccess)

    def run(self, origin, succeeds):
        return _wzalgorithm.REvol_run(self, origin, succeeds)
    __swig_destroy__ = _wzalgorithm.delete_REvol

# Register REvol in _wzalgorithm:
_wzalgorithm.REvol_swigregister(REvol)

def REvol_pt1(y, u, t):
    return _wzalgorithm.REvol_pt1(y, u, t)

def REvol_agePopulation(population):
    return _wzalgorithm.REvol_agePopulation(population)

class REvolIndividual(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    parameters = property(_wzalgorithm.REvolIndividual_parameters_get, _wzalgorithm.REvolIndividual_parameters_set)
    scatter = property(_wzalgorithm.REvolIndividual_scatter_get, _wzalgorithm.REvolIndividual_scatter_set)
    timeToLive = property(_wzalgorithm.REvolIndividual_timeToLive_get, _wzalgorithm.REvolIndividual_timeToLive_set)
    restrictions = property(_wzalgorithm.REvolIndividual_restrictions_get, _wzalgorithm.REvolIndividual_restrictions_set)

    def __init__(self, *args):
        _wzalgorithm.REvolIndividual_swiginit(self, _wzalgorithm.new_REvolIndividual(*args))

    def age(self):
        return _wzalgorithm.REvolIndividual_age(self)

    def isBetterThan(self, other):
        return _wzalgorithm.REvolIndividual_isBetterThan(self, other)

    @staticmethod
    def isIndividual1Better(i1, i2):
        return _wzalgorithm.REvolIndividual_isIndividual1Better(i1, i2)

    def __eq__(self, other):
        return _wzalgorithm.REvolIndividual___eq__(self, other)

    def __ne__(self, other):
        return _wzalgorithm.REvolIndividual___ne__(self, other)

    def __lt__(self, other):
        return _wzalgorithm.REvolIndividual___lt__(self, other)
    __swig_destroy__ = _wzalgorithm.delete_REvolIndividual

# Register REvolIndividual in _wzalgorithm:
_wzalgorithm.REvolIndividual_swigregister(REvolIndividual)

def REvolIndividual_isIndividual1Better(i1, i2):
    return _wzalgorithm.REvolIndividual_isIndividual1Better(i1, i2)

class ParticleSwarmOptimization(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    DEFAULT_SWARM_SIZE = _wzalgorithm.ParticleSwarmOptimization_DEFAULT_SWARM_SIZE

    def __init__(self):
        _wzalgorithm.ParticleSwarmOptimization_swiginit(self, _wzalgorithm.new_ParticleSwarmOptimization())

    def swarmSize(self, *args):
        return _wzalgorithm.ParticleSwarmOptimization_swarmSize(self, *args)

    def maxIterations(self, *args):
        return _wzalgorithm.ParticleSwarmOptimization_maxIterations(self, *args)

    def lowerBoundary(self, *args):
        return _wzalgorithm.ParticleSwarmOptimization_lowerBoundary(self, *args)

    def upperBoundary(self, *args):
        return _wzalgorithm.ParticleSwarmOptimization_upperBoundary(self, *args)

    def neighbors(self, particleIndex, swarmSize):
        return _wzalgorithm.ParticleSwarmOptimization_neighbors(self, particleIndex, swarmSize)

    def bestPreviousBestPosition(self, neighborhood):
        return _wzalgorithm.ParticleSwarmOptimization_bestPreviousBestPosition(self, neighborhood)

    def createSwarm(self, dimension, evaluator):
        return _wzalgorithm.ParticleSwarmOptimization_createSwarm(self, dimension, evaluator)

    def run(self, dimension, evaluator):
        return _wzalgorithm.ParticleSwarmOptimization_run(self, dimension, evaluator)
    __swig_destroy__ = _wzalgorithm.delete_ParticleSwarmOptimization

# Register ParticleSwarmOptimization in _wzalgorithm:
_wzalgorithm.ParticleSwarmOptimization_swigregister(ParticleSwarmOptimization)
cvar = _wzalgorithm.cvar
ParticleSwarmOptimization.C = _wzalgorithm.cvar.ParticleSwarmOptimization_C
ParticleSwarmOptimization.W = _wzalgorithm.cvar.ParticleSwarmOptimization_W

class ParticleSwarmOptimizationParticle(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    bestFitness = property(_wzalgorithm.ParticleSwarmOptimizationParticle_bestFitness_get, _wzalgorithm.ParticleSwarmOptimizationParticle_bestFitness_set)
    currentFitness = property(_wzalgorithm.ParticleSwarmOptimizationParticle_currentFitness_get, _wzalgorithm.ParticleSwarmOptimizationParticle_currentFitness_set)
    bestPosition = property(_wzalgorithm.ParticleSwarmOptimizationParticle_bestPosition_get, _wzalgorithm.ParticleSwarmOptimizationParticle_bestPosition_set)
    currentPosition = property(_wzalgorithm.ParticleSwarmOptimizationParticle_currentPosition_get, _wzalgorithm.ParticleSwarmOptimizationParticle_currentPosition_set)
    velocity = property(_wzalgorithm.ParticleSwarmOptimizationParticle_velocity_get, _wzalgorithm.ParticleSwarmOptimizationParticle_velocity_set)

    def __lt__(self, rhs):
        return _wzalgorithm.ParticleSwarmOptimizationParticle___lt__(self, rhs)

    def __init__(self):
        _wzalgorithm.ParticleSwarmOptimizationParticle_swiginit(self, _wzalgorithm.new_ParticleSwarmOptimizationParticle())
    __swig_destroy__ = _wzalgorithm.delete_ParticleSwarmOptimizationParticle

# Register ParticleSwarmOptimizationParticle in _wzalgorithm:
_wzalgorithm.ParticleSwarmOptimizationParticle_swigregister(ParticleSwarmOptimizationParticle)

class ParticleSwarmOptimizationResult(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    bestParticle = property(_wzalgorithm.ParticleSwarmOptimizationResult_bestParticle_get, _wzalgorithm.ParticleSwarmOptimizationResult_bestParticle_set)
    iterationsUsed = property(_wzalgorithm.ParticleSwarmOptimizationResult_iterationsUsed_get, _wzalgorithm.ParticleSwarmOptimizationResult_iterationsUsed_set)

    def __init__(self):
        _wzalgorithm.ParticleSwarmOptimizationResult_swiginit(self, _wzalgorithm.new_ParticleSwarmOptimizationResult())
    __swig_destroy__ = _wzalgorithm.delete_ParticleSwarmOptimizationResult

# Register ParticleSwarmOptimizationResult in _wzalgorithm:
_wzalgorithm.ParticleSwarmOptimizationResult_swigregister(ParticleSwarmOptimizationResult)


def __lshift__(*args):
    return _wzalgorithm.__lshift__(*args)
class REvolSuccessPredicate(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, evaluatorObject):
        _wzalgorithm.REvolSuccessPredicate_swiginit(self, _wzalgorithm.new_REvolSuccessPredicate(evaluatorObject))
    __swig_destroy__ = _wzalgorithm.delete_REvolSuccessPredicate

    def __call__(self, i):
        return _wzalgorithm.REvolSuccessPredicate___call__(self, i)

# Register REvolSuccessPredicate in _wzalgorithm:
_wzalgorithm.REvolSuccessPredicate_swigregister(REvolSuccessPredicate)


