# Paper Code Release

## Environment Configuration

### Semi-Supervised Framework

The environment for the main semi-supervised framework is configured according to the instructions in the [DSL repository](https://github.com/chenbinghui1/dsl). Please follow the setup instructions provided there to install the necessary dependencies and configure your environment.

### OOD Detector

For the OOD detector, both the environment and the dataset are configured as detailed in the [OpenDet2 repository](https://github.com/csuhan/opendet2). Refer to that repository for guidance on environment setup, dependency installation, and dataset preparation.

## Pre-trained Model Weights

The repository includes the trained model weights produced by our experiments. Please ensure that you place the model weights in the correct directory as expected by the testing scripts.

## Testing

To test the model, run the following command:

```bash
python ./test.py
