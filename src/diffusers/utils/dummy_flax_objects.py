# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..utils import DummyObject, requires_backends


class FlaxControlNetModel(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxModelMixin(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxUNet2DConditionModel(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxAutoencoderKL(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxDiffusionPipeline(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxDDIMScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxDDPMScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxDPMSolverMultistepScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxEulerDiscreteScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxKarrasVeScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxLMSDiscreteScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxPNDMScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxSchedulerMixin(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])


class FlaxScoreSdeVeScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])