# An Updated Testing Framework

Below is a summary of the testing framework proposed for sparseml. 

## Existing Tests

### Integration Tests

Existing integration tests are rewritten such that all values relevant to the particular
test case are read from a config file, as opposed to hardcoded values in the test case
itself or through overloaded pytest fixtures. Each config file should include one 
combination of relevant parameters that are needed to be tested for that particular
integration test. Each config file must at least have the values defined by the 
`TestConfig` dataclass found under `tests/data`. These values include the `cadence`
(weekly, commit, or nightly) and the `test_type` (sanity, smoke, or regression) for the
particular test case. While the `test_type` is currently using a config value, we can
expand it to use pytest markers instead. An example of this updated approach can be
found in the export test case, `test_generation_export.py`

### Unit Tests

Unit tests are not changed significantly however, can be adapted to use the 
`unittest.TestCase` base class. While this is not necessary to be used, it does
seem like `unittest` provides overall greater readability compared to normal pytest
tests. There is also the case where we can use both pytest and unittest for our test
cases. This is not uncommon and also what transformers currently does. An example of 
an updated test can be in the `test_export_data_new.py` test file. A note about using
`unittest` is that it requires us to install the `parameterized` package for
decorating test cases.

## Custom Testing

For the purpose of custom integration testing, two new workflows are now enabled 

1. **Custom Script Testing**: Users can test their custom python script which is not 
required to follow any specific structure. All asserts in the script will be validated
2. **Custom Testing Class**: For slightly more structure, users can write their own
testing class. The only requirement is that this testing class inherits from the base
class `CustomTestCase` which can be found under `tests/custom_test`.

To enable custom integration testing for any of the cases above, a test class must be
written which inherits from `CustomIntegrationTest` under tests/custom_test. Within this
class, two paths can be defined: `custom_scripts_directory` which points to the
directory containing all the custom scripts which are to be tested and 
`custom_class_directory` which points to the directory containing all the custom test
classes.

Similar to the non-custom integration testing, each custom integration test script or
test class must include a .yaml file which includes 3 values 
(defined by the `CustomTestConfig` dataclass found under `tests/data`): 
`test_type` which indicates if the test is a sanity, smoke or regression test, 
`cadence`: which dictates how often the test case runs (on commit, weekly, or nightly), 
and the `script_path` which lists the name of the custom testing script or custom test
class within the directory. 

An example of an implementation of the `CustomIntegrationTest` can be found under
`tests/examples`

## Additional markers and decorators 

- New markers are added in to mark tests as `unit`, `integration`, `smoke`, and `custom` 
tests allowing us to run a subset of the tests when needed
- Two new decorators are added in to check for package and compute requirements. If
the requirements are not met, the test is skipped. Currently, `requires_torch` and 
`requires_gpu` are added in and can be found under `testing_utils.py`

## Testing Targets 

### Unit Testing Targets:
- A unit test should be written for every utils, helper, or static function. 
    - Test cases should be written for all datatype combinations that the function takes as input 
    - Can have `smoke` tests but focus should be on `sanity`

### Integration Testing Targets:
- An integration test should be written for every cli pathway that is exposed through `setup.py`
    - All cli-arg combinations should be tested through a `smoke` check 
    (all may be overkill but ideally we're covering beyond the few important combinations)
    - All **important** cli-arg combinations should be covered through either a `sanity`
    check or a `regression` check
        - A small model should be tested through a `sanity` check  
        - All other larger models should be tested through `regression` test types
   
- An integration test should be written for every major/critical module 
    - All arg combinations should be tested through a `smoke` check
    (all may be overkill but ideally we're covering beyond the few important combinations)
    - All **important** arg combinations should be covered through either a `sanity`
    check or a `regression` check
        - A small model should be tested through a `sanity` check  
        - All other larger models should be tested through `regression` test types

## End-to-end Testing Targets:
- Tests cascading repositories (sparseml --> vLLM) but will become more prominent as our
docker containers are furhter solidified. Goal would be to emulate common flows users
may follow

## Cadence
- Ideally, large models and `regression` tests should be tested on a nightly cadence while
unit tests and `sanity` test should be tested on a per commit basis