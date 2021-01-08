# Contributing to SparseML

If you’re reading this, hopefully we have piqued your interest to take the next step. Join us and help make SparseML even better! As a contributor, here are some community guidelines we would like you to follow:

 - [Code of Conduct](#code-of-conduct)
 - [Ways to Contribute](#ways-to-contribute)
 - [Bugs and Feature Requests](#bugs-and-feature-requests)
 - [Question or Problem?](#question-or-problem)
 - [Developing SparseML](#developing-sparseml)
 - [How to Contribute](#how-to-contribute)

 ## Code of Conduct
Help us keep the software inclusive. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) in order to promote an environment that is friendly, fair, respectful, and safe. We want to inspire collaboration, innovation, and fun!

 ## Ways to Contribute

Whether you’re a newbie, dabbler, or expert, we appreciate you jumping in.

 - Resolve open issues with the existing code;
 - Contribute to documentation, examples, tutorials;
 - Submit bugs, propose features;
 - Answer open discussion topics;
 - Spread the word about SparseML (and if you feel inclined, @mention Neural Magic as we love hearing about it);
 - Teach and empower others. This is the way!

 ## Bugs and Feature Requests
 - Check out [GitHub Issues](https://github.com/neuralmagic/sparseml/issues) for reporting bugs and feature requests

Prior to creating an issue, please search through existing issues so that you are not creating duplicate ones. 

If it’s a bug, be sure to provide where applicable, the s.t.r. = steps to reproduce including any code snippets, screenshots, traceback if an exception was raised, log content, and/or sample model files. 

If proposing a feature request, share the problem you’re trying to solve, community benefits, and other relevant details to support your proposal.

If you want to contribute a model, start with sharing a brief introduction at [GitHub Discussions under Ideas](https://github.com/neuralmagic/sparseml/discussions/categories/ideas). Fellow Neural Magic engineers can work with your further to take next steps.

 ## Question or Problem
 - Post a topic to [GitHub Discussions](https://github.com/neuralmagic/sparseml/discussions/) for all other questions including support or how to contribute

Don’t forget to search through existing discussions to avoid duplicating posts! Thanks!

## Developing SparseML

_*Edit the following topics as it relates to the developing software of this repo and remove this comment; the following is placeholder content and may not be applicable or there may be missing sections.*_

### Building and running SparseML locally

SparseML is OS-agnostic. Requires Python 3.6 or higher. 
```python
$ pip install sparseml

```
This will do [list out concise summary of what these command will do and what the use can expect to happen once installed].
  
To build and run SparseML locally, run the following commands:
```shell
$ <command 1>
$ <command 2>
$ <command n>
```

This will do [list out concise summary of what these command will do and what the use can expect to happen once installed].


### Testing changes locally

To build and test changes locally, run the following commands:

```shell
$ <command n>
```

### Testing Docker image

See comments at the top of the [Docker compose file in the root of the project](docker-compose.yml) for instructions
on how to build and run Docker images.

## How to Contribute

### Guidelines for Contributing Code, Examples, Documentation

Code changes are submitted via a pull request (PR). Confirm first by searching the issues or PRs that someone is not already working on the same task. When submitting a PR use the following guidelines:

* Follow the style guide below
* Add/update documentation appropriately for the change you are making. For more information, see the [docs readme](docs/readme.md).
* ??? Non-trivial changes should include unit tests covering the new functionality and potentially [function tests](path/to/src/test/resources/name_here/README.md).
* ??? All syntax changes and enhancements should come with appropriate [function tests](path/to/src/test/resources/name_here/README.md).
* ??? Bug fixes should include a unit test or integration test potentially [function tests](path/to/src/test/resources/query-validation-tests/README.md) proving the issue is fixed.
* Try to keep pull requests short and submit separate ones for unrelated features.
* Keep formatting changes in separate commits to make code reviews easier and distinguish them from actual code changes.

#### Code Style

The project uses [GoogleStyle](https://google.github.io/styleguide/pyguide.html) code formatting.

The project also uses immutable types where ever possible. If adding new type(s), ideally make them immutable.

#### Static Code Analysis

_Remove section if not needed._

#### Commit Message Format

The project uses [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/) for commit messages
in order to aid in automatic generation of changelogs. As described in the Conventional Commmits specification,
commit messages should be of the form:

    <type>[optional scope]: <description>

    [optional body]

    [optional footer]

where the `type` is one of
 * "fix": for bug fixes
 * "feat": for new features
 * "refactor": for refactors
 * "test": for test-only changes
 * "docs": for docs-only changes
 * "revert": for reverting other changes
 * "perf", "style", "build", "ci", or "chore": as described in the [Angular specification](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type) for Conventional Commits.

##### Breaking Changes

A commit is a "breaking change" if users should expect different behavior from an existing workflow
as a result of the change. Examples of breaking changes include deprecation of existing configs or APIs,
changing default behavior of existing configs or query semantics, or the renaming of exposed JMX metrics.
Breaking changes must be called out in commit messages, PR descriptions, and upgrade notes:

 * Commit messages for breaking changes must include a line (in the optional body or footer)
   starting with "BREAKING CHANGE: " followed by an explanation of what the breaking change was.
   For example,

 *  * feat: allow provided config object to extend other configs
     
 *  * BREAKING CHANGE: `extends` key in config file is now used for extending other config files  
     
 * The breaking change should similarly be called out in the PR description.
   This description will be copied into the body of the final commit merged into the repo,
   and picked up by our automatic changelog generation accordingly.

 * [Upgrade notes](https://github.com/neuralmagic/sparseml/blob/master/docs/installation/upgrading.rst)
   should also be updated as part of the same PR.

##### Commitlint

This project has [commitlint](https://github.com/conventional-changelog/commitlint) configured
to ensure that commit messages are of the expected format.
To enable commitlint, simply run `npm install` from the root directory of the sparseml repo
(after [installing `npm`](https://www.npmjs.com/get-npm).)
Once enabled, commitlint will reject commits with improperly formatted commit messages.

### GitHub Workflow

1. Fork the `neuralmagic/sparseml` repository into your GitHub account: https://github.com/neuralmagic/sparseml/fork.

2. Clone your fork of the GitHub repository, replacing `<username>` with your GitHub username.

   Use ssh (recommended):

   ```bash
   git clone git@github.com:<username>/sparseml.git
   ```

   Or https:

   ```bash
   git clone https://github.com/<username>/sparseml.git
   ```

3. Add a remote to keep up with upstream changes.

   ```bash
   git remote add upstream https://github.com/neuralmagic/sparseml.git
   ```

   If you already have a copy, fetch upstream changes.

   ```bash
   git fetch upstream
   ```

4. Create a feature branch to work in.

   ```bash
   git checkout -b feature-xxx remotes/upstream/master
   ```

5. Work in your feature branch.

   ```bash
   git commit -a
   ```

6. Periodically rebase your changes

   ```bash
   git pull --rebase
   ```

7. When done, combine ("squash") related commits into a single one

   ```bash
   git rebase -i upstream/master
   ```

   This will open your editor and allow you to re-order commits and merge them:
   - Re-order the lines to change commit order (to the extent possible without creating conflicts)
   - Prefix commits using `s` (squash) or `f` (fixup) to merge extraneous commits.

8. Submit a pull-request

   ```bash
   git push origin feature-xxx
   ```

   Go to your fork main page

   ```bash
   https://github.com/<username>/sparseml
   ```

   If you recently pushed your changes GitHub will automatically pop up a `Compare & pull request` button for any branches you recently pushed to. If you click that button it will automatically offer you to submit your pull-request to the `neuralmagic/sparseml` repository.

   - Give your pull-request a meaningful title that conforms to the Conventional Commits specification
     as described [above](#commit-message-format) for commit messages.
     You'll know your title is properly formatted once the `Semantic Pull Request` GitHub check
     transitions from a status of "pending" to "passed".
   - In the description, explain your changes and the problem they are solving. Be sure to also call out
     any breaking changes as described [above](#breaking-changes).

9. Addressing code review comments

   Repeat steps 5. through 7. to address any code review comments and rebase your changes if necessary.

   Push your updated changes to update the pull request

   ```bash
   git push origin [--force] feature-xxx
   ```

   `--force` may be necessary to overwrite your existing pull request in case your
  commit history was changed when performing the rebase.

   Note: Be careful when using `--force` since you may lose data if you are not careful.

   ```bash
   git push origin --force feature-xxx
   ```