# SparseML Integration Status Tooling
This directory contains a status page across all SparseML integration statuses
as well as tooling to generate status pages

## Status Pages
The aggregated status page is found in this directory in `STATUS.MD`.
Individual integration status pages are found in specific integration directories
with the extension `.status.md`.

## Adding a New Status Page
To create a status page for a new integration:
* Copy `status_template.status.yaml` into a file named `{integration_name}.status.yaml` in the same directory as the integration in `src/`
* All fields will default to `n`. Follow the key to update features to the proper status 
* Run `make status` to update and create the auto generated markdown file
* Check in to git and open a PR

## Updating an Existing Status Page
To update an existing status page due to a feature change:
* Locate the integration's `{integration}.status.yaml` file in the integration directory in SparseML source
* Update the line item(s) of the changed features according to the status key
* run `make status` to update the auto generated markdown files
* Check in to git and open a PR

## Add a new Feature to a Table or new Table
To add a new feature to a table or a new table:
* Open `status/integration_status.py` and either create a new `FeatureStatus` Field in an existing table or create a table
* Follow existing conventions
* Update all existing `.status.yaml` files nested under `src/` to include the new feature(s)
* Run `make status` (if any errors occur, make sure the old configs were properly updated)
* Check `git diff` to make sure the new feature(s) had the intended effect on the auto generated markdown files and auto generated template
* Check in to git and open a PR
