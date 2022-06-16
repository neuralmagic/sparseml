# Integrations Testing
This directory holds the automated testing suite for `SparseML integrations`. The list of current supported integrations is:

- HuggingFace Transformers
- Ultralytics Yolov5 (Object Detection)
- Image Classification

The testing suite for each integration is contained in its respective directory and is based off the components in the `tests/integrations` and the patterns laid out in this document.

## Required Components
The required components to implement testing for an integration are listed here. For the manager and tester classes, see the in-code documentation for their base classes. For the arg classes, look at one of the existing implementations. 

**Integration Manager Class:** Inherited from `BaseIntegrationManager`, this class handles the setup, running, and teardown of integration components. 

**Integration Tester Class:** Inherited from `BaseIntegrationTester`, this class contains the tests to be executed after a scenario run.

**Arg Classes:** Set of pydantic classes which contain all the CLI args for each stage of a run: train, export, and deploy (as applicable).  

**Scenario YAML Configs:** Each YAML config defines a scenario to be run. Upon test time, the testing suite is parameterized over the set of config files which match the current cadence. See more in [Scenario Yaml Config Structure](#Scenario-Yaml-Config-Structure).

## Scenario YAML Config Structure
Each config file contains one scenario composed of one or more stages. Currently supported stages are `train`, `export`, and `deploy` (WIP). This section will cover all the supported fields for a config file.

## Template/Example Config
A template config file is outlined below. It shows all supported fields, but not all displayed fields are required. In addition, values and some keys displayed are example values which may not hold for specific integrations. Please see the section for each field for more details.
```yaml
cadence: "commit"
abridged_mode: True
train:
    pre_args: CUDA_VISIBLE_DEVICES=1
    command_args:
        weights: path/to/weights
        epochs: 10
        data: path/to/data
    test_args: 
        target_name: "map0.5"
        target_mean: 0 
        target_std: 1
export:
    pre_args: sample_pre_arg
    command_args:
        weight_path: path/to/weights
        dynamic: True
    test_args:
        target_model: zoo/sub/to/model
deploy:
    pre_args: sample_pre_arg
    command_args:
        model_path: path/to/model
        data: path/to/data
    test_args:
        test_arg: test_value
```

## Config Fields
**cadence:** [Required] Defines how often the test should be run. Supported values are `commit`, `nightly`, and `weekly`. Upon testing, config files with the desired cadence will be fetched, ran, and tested.

**abridged_mode:** [Optional] When true, the length of the run will be automatically shortened to a standardized abridged version. e.g. for training, a default, low value of maximum training and eval steps will be set. This allows for testing of long runs on a higher cadence. Naturally, metrics can not be tested for abridged runs. Defaults to False when not set

**Stage Name:** [>1 Required] Supported stage names are `train`, `export`, and `deploy` [WIP]. None are individually required, but at least one must be present in the config. 

**pre_args:** [WIP][Optional] Allows for the addition of CLI commands to prepend to the run command. The most common use case is for setting environment variables.

**command_args:** [Optional/Required] The arguments to be passed to each stage of the run. These arguments are converted into CLI commands for runtime (example below). Supported args are defined by the `arg class` implementations for the integration. Thus if the corresponding class defines required arguments, then the `command_args` field is required. Field names provided in the config that don't correspond to supported arguments in the code are flagged with an error to prevent user error. Example of arg conversions from config to CLI command <br>
`'epochs: 5' -> '--epochs 5'` <br>
`'dynamic: True' -> '--dynamic'`

**test_args:** These arguments provide information to the tests on what quantities and qualities of the run output to test. These can integration specific, but right now the following set of `test args` are supported across the integrations:
- Train
    - `target_name`: Name of the metric to test
    - `target_mean`: Mean value of the named metric
    - `target_std`: Acceptable range of deviation from the target mean for the named metric
- Export
    - `target_model`: zoo stub for a model to test the onnx model produced by the run against

