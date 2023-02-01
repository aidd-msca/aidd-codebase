"""This module contains the registries for the AIDD codebase."""

from registry_factory.factory import Factory


class AIDD(Factory):
    DataRegistry = Factory.create_registry(shared=False)

    ModuleRegistry = Factory.create_registry(shared=True)
    ModelRegistry = Factory.create_registry(shared=True)

    MetricRegistry = Factory.create_registry(shared=False)
