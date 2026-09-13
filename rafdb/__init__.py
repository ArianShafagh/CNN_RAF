"""RAF-DB Basic facial expression recognition with a fine-tuned ResNet-18.

Sub-packages, in the order you would use them:

    rafdb.core       config, datasets, model — imported by everything else
    rafdb.pipeline   train, evaluate, export — the stages that produce a model
    rafdb.reporting  figures and multi-seed statistics
    rafdb.deploy     live inference on the exported ONNX graph

Run a stage as a module from the project root, for example:

    python -m rafdb.pipeline.train --seed 42

or run the whole pipeline with `python3 run_all.py`.
"""
