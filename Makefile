BUILDDIR := $(PWD)
DOCDIR := docs
PACKAGE := neuralmagicML-python.tar.gz
LOGSDIR := test_logs
.ONESHELL:
SHELL := /bin/bash


install:
	python3 -m venv .venv;
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

version_testing:
	@source .venv/bin/activate;
	for tensorflow_version in 1.8 1.9 1.10 1.11 1.12 1.13 1.14 1.15; do \
		tensorflow="tensorflow==$$tensorflow_version.*"; \
		pip3 install "$$tensorflow"; \
		$(MAKE) run_tests; \
	done;
	for torch_version in 1.1 1.2 1.3 1.4; do \
		torch="torch==$$torch_version.*"; \
		if (( $$(echo "$$torch_version < 1.1" | bc -l) )); then \
			torchvision="torchvision==0.2.*"; \
		elif (( $$(echo "$$torch_version < 1.2" | bc -l) )); then \
			torchvision="torchvision==0.3.0"; \
		elif (( $$(echo "$$torch_version < 1.3" | bc -l) )); then \
			torchvision="torchvision==0.4.0"; \
		elif (( $$(echo "$$torch_version < 1.4" | bc -l) )); then \
			torchvision="torchvision==0.4.2"; \
		else \
			torchvision="torchvision==0.5.0"; \
		fi; \
		pip3 install "$$torch" "$$torchvision"; \
		$(MAKE) run_tests; \
	done; \

run_tests:
	pytest . 2>&1 | tee "$(LOGSDIR)/$$(pip3 freeze | grep tensorflow==)_$$(pip3 freeze | grep torch==)_$$(pip3 freeze | grep torchvision==).log";
	
clean:
	rm -f $(PACKAGE);
	rm -rf __pycache__ .pytest_cache .venv test_logs;
	rm -rf .vscode build dist neuralmagicML.egg-info;
	