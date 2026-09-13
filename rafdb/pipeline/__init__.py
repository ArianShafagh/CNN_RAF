"""The stages that turn data into a deployable model.

    train     fine-tune ResNet-18, select on validation macro-F1
    evaluate  score a split and write the report, metrics JSON and figures
    export    convert the checkpoint to ONNX and verify it against PyTorch
"""
