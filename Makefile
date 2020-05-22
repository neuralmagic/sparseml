BUILDDIR := $(PWD)
DOCDIR := docs
PACKAGE := neuralmagicML-python.tar.gz
LOGSDIR := test_logs
.ONESHELL:
SHELL := /bin/bash

tensorflow_testing: export NM_ML_SKIP_TENSORFLOW_TESTS = False
tensorflow_testing: export NM_ML_SKIP_PYTORCH_TESTS = True

pytorch_testing: export NM_ML_SKIP_TENSORFLOW_TESTS = True
pytorch_testing: export NM_ML_SKIP_PYTORCH_TESTS = False

install:
	python3 -m venv .venv || virtualenv --python=$$(which python3) .venv;
	git submodule update;
	source .venv/bin/activate;
	pip3 install --upgrade pip;
	pip3 install .;
	pip3 install sphinxcontrib-apidoc rinohtype;
	[ ! -d $(LOGSDIR) ] && mkdir $(LOGSDIR) || echo "$(LOGSDIR) already exists";

doc:
	source .venv/bin/activate;
	sphinx-apidoc -o docs/source/ neuralmagicML/;
	cd $(DOCDIR) && $(MAKE) html && sphinx-build -b rinoh source _build/rinoh;
	cd $(BUILDDIR);

package:
	$(MAKE) doc;
	git submodule update;
	python3 --tar-name $$PACKAGE;

tensorflow_testing:
	@source .venv/bin/activate;
	for tensorflow_version in 1.8 1.9 1.10 1.11 1.12 1.13 1.14 1.15; do \
		if [ -z $$NM_ML_GPU_TESTS ]; then \
			pip3 install "tensorflow==$$tensorflow_version.*"; \
		else \
			pip3 install "tensorflow==$$tensorflow_version.*" "tensorflow-gpu==$$tensorflow_version.*"; \
		fi; \
		pytest . 2>&1 | tee "$(LOGSDIR)/$$(pip3 freeze | grep tensorflow==).log"; \
	done;

pytorch_testing:
	@source .venv/bin/activate;
	for torch_version in 1.1 1.2 1.3 1.4 1.5; do \
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
		else \
			torchvision="torchvision==0.6.0"; \
		fi; \
		pip3 install "$$torch" "$$torchvision"; \
		pytest . 2>&1 | tee "$(LOGSDIR)/$$(pip3 freeze | grep torch==).log"; \
	done;

test_latest:
	@source .venv/bin/activate;
	pip3 install tensorflow==1.15 torch==1.5 torchvision==0.6.0;
	pytest . 2>&1 | tee "$(LOGSDIR)/$$(pip3 freeze | grep torch==)_$$(pip3 freeze | grep tensorflow==).log";

version_testing:
	$(MAKE) test_latest;
	$(MAKE) tensorflow_testing;
	$(MAKE) pytorch_testing;

python_version_testing:
	@for python_version in $$(pyenv install --list | grep "3\.[678]" ); do \
		[ -z $$(pyenv versions | grep "$$python_version") ] && pyenv install "$$python_version"; \
		[ -d "$$HOME/.pyenv/versions/pyenv_$$python_version" ] || pyenv virtualenv "$$python_version" "pyenv_$$python_version"; \
		source "$$HOME/.pyenv/versions/pyenv_$$python_version/bin/activate"; \
		pip3 install .; \
		pytest . 2>&1 | tee "$(LOGSDIR)/python==$$python_version.log"; \
	done;

clean:
	rm -f $(PACKAGE);
	rm -rf __pycache__ .pytest_cache .venv test_logs;
	rm -rf .vscode build dist neuralmagicML.egg-info;
	