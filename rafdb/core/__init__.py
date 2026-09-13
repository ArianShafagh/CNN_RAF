"""Shared foundation: configuration, data pipeline, and model definition.

Nothing here runs a stage; these modules are imported by pipeline, reporting
and deploy so that all four read the same hyperparameters and class ordering.
"""
