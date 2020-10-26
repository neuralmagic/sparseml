BUILDDIR := $(PWD)
DOCDIR := docs
PACKAGE := neuralmagicML-python.tar.gz
LOGSDIR := test_logs
.ONESHELL:
SHELL := /bin/bash
FAILURE_LOG := "$(LOGSDIR)/failures.log"

tensorflow_testing: export NM_ML_SKIP_TENSORFLOW_TESTS = False
tensorflow_testing: export NM_ML_SKIP_PYTORCH_TESTS = True

pytorch_testing: export NM_ML_SKIP_TENSORFLOW_TESTS = True
pytorch_testing: export NM_ML_SKIP_PYTORCH_TESTS = False

install:
	python3 -m venv .venv || virtualenv --python=$$(which python3) .venv;
	source .venv/bin/activate && pip3 install --upgrade pip && pip3 install .;
	pip3 install sphinxcontrib-apidoc rinohtype;

doc:
	source .venv/bin/activate;
	sphinx-apidoc -o docs/source/ neuralmagicML/;
	cd $(DOCDIR) && $(MAKE) html;
	cd $(BUILDDIR);

package:
	$(MAKE) doc;
	python3 create_package.py --tar-name $(PACKAGE);

tensorflow_testing:
	@source .venv/bin/activate;
	for tensorflow_version in 1.8 1.9 1.10 1.11 1.12 1.13 1.14 1.15; do \
		if [ -z $$NM_ML_GPU_TESTS ]; then \
			pip3 install . "tensorflow==$$tensorflow_version.*" "tensorboard==$$tensorflow_version.*" &> /dev/null; \
		else \
			pip3 install . "tensorflow==$$tensorflow_version.*" "tensorboard==$$tensorflow_version.*" "tensorflow-gpu==$$tensorflow_version.*" &> /dev/null; \
		fi; \
		echo "Running tests with $$(pip3 freeze | grep tensorflow==)"; \
		pytest .; \
		if [[ -f "$(FAILURE_LOG)" ]]; then \
			cp "$(FAILURE_LOG)" "$(LOGSDIR)/$$(pip3 freeze | grep tensorflow==).log"; \
		fi \
	done;

pytorch_testing:
	@source .venv/bin/activate;
	for torch_version in 1.1 1.2 1.3 1.4 1.5 1.6; do \
		torch="torch==$$torch_version.*"; \
		if (( $$(echo "$$torch_version < 1.1" | bc -l) )); then \
			torchvision="torchvision==0.2.*"; \
		elif (( $$(echo "$$torch_version < 1.2" | bc -l) )); then \
			torchvision="torchvision==0.3.0"; \
		elif (( $$(echo "$$torch_version < 1.3" | bc -l) )); then \
			torchvision="torchvision==0.4.0"; \
		elif (( $$(echo "$$torch_version < 1.4" | bc -l) )); then \
			torchvision="torchvision==0.4.2"; \
		elif (( $$(echo "$$torch_version < 1.5" | bc -l) )); then \
			torchvision="torchvision==0.5.0"; \
		elif (( $$(echo "$$torch_version < 1.6" | bc -l) )); then \
			torchvision="torchvision==0.6.0"; \
		else \
			torchvision="torchvision==0.7.0"; \
		fi; \
		pip3 install . "$$torch" "$$torchvision" &> /dev/null; \
		echo "Running tests with $$(pip3 freeze | grep torch==)"; \
		pytest .; \
		if [[ -f "$(FAILURE_LOG)" ]]; then \
			cp "$(FAILURE_LOG)" "$(LOGSDIR)/$$(pip3 freeze | grep torch==).log"; \
		fi \
	done;

test_latest:
	@source .venv/bin/activate;
	pip3 uninstall -y tensorflow torch torchvision tensorboard &> /dev/null;
	pip3 install . torch==1.6 torchvision==0.7 tensorflow==1.15 &> /dev/null;
	if [ ! -z $$NM_ML_GPU_TESTS ]; then \
		tfversion="$$(pip3 freeze | grep tensorflow==)";\
		tfgpuversion=$$(echo "$${tfversion/tensorflow/tensorflow-gpu}");\
		pip3 install "$$tfgpuversion"; \
	fi;
	echo "Running tests with $$(pip3 freeze | grep torch==) and $$(pip3 freeze | grep tensorflow==)";
	pytest .;
	if [[ -f "$(FAILURE_LOG)" ]]; then \
		cp "$(FAILURE_LOG)" "$(LOGSDIR)/$$(pip3 freeze | grep torch==)_$$(pip3 freeze | grep tensorflow==).log"; \
	fi

version_testing:
	rm -fr "$(LOGSDIR)";
	mkdir "$(LOGSDIR)";
	$(MAKE) test_latest;
	$(MAKE) tensorflow_testing;
	$(MAKE) pytorch_testing;
	if [[ ! -d "$(LOGSDIR)" ]] || [[ -z "$$(ls -A '$(LOGSDIR)')" ]]; then \
		echo "No version errors found"; \
	else \
		exit 1; \
	fi;

python_version_testing:
	@rm -fr "$(LOGSDIR)";
	mkdir "$(LOGSDIR)";
	for python_version in $$(pyenv install --list | grep " 3\.[5678]" ); do \
		echo "Testing python version $$python_version"; \
		[ -z "$$(pyenv versions | grep "$$python_version")" ] && env PYTHON_CONFIGURE_OPTS="--enable-framework CC=clang" pyenv install "$$python_version"; \
		[ -d "$$HOME/.pyenv/versions/pyenv_$$python_version" ] || pyenv virtualenv "$$python_version" "pyenv_$$python_version"; \
		source "$$HOME/.pyenv/versions/pyenv_$$python_version/bin/activate"; \
		
		pip3 install . &> /dev/null; \
		echo "Running tests with python version $$(python_version)"; \
		pytest .; \

		if [[ -f "$(FAILURE_LOG)" ]]; then \
			cp "$(FAILURE_LOG)" "$(LOGSDIR)/python==$$python_version.log"; \
		fi \
	done;
	if [[ ! -d "$(LOGSDIR)" ]] || [[ -z "$$(ls -A '$(LOGSDIR)')" ]]; then \
		echo "No version errors found"; \
	else \
		exit 1; \
	fi;

clean:
	rm -f $(PACKAGE);
	rm -fr .pytest_cache .venv $(LOGSDIR) .vscode;
	rm -fr docs/_build docs/build;
	rm -fr tensorboard;
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -fr;
	find . | grep .rst | xargs rm -fr;
	