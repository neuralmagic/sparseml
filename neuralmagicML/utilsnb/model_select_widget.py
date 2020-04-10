"""
Code related to model repo selection display in a jupyter notebook using ipywidgets
"""

from typing import List, Union
import ipywidgets as widgets

from neuralmagicML.utils import available_models, filter_model, RepoModel
from neuralmagicML.utilsnb.helpers import format_html


__all__ = ["ModelSelectWidgetContainer"]


class _FilterWidget(object):
    def __init__(self, all_models: List[RepoModel]):
        self._all_models = all_models
        self._domains = self._init_domains()
        self._datasets = self._init_datasets()

        self._recal_checkbox = widgets.Checkbox(value=False, indent=False)
        self._domains_dropdown = widgets.Dropdown(
            options=self._domains, value=self._domains[0]
        )
        self._datasets_dropdown = widgets.Dropdown(
            options=self._datasets, value=self._datasets[0]
        )
        self._setup_hooks()

        self.container = widgets.VBox(
            (
                widgets.HTML(format_html("Filters", header="h4")),
                widgets.HBox(
                    (widgets.HTML(format_html("Domains:")), self._domains_dropdown,)
                ),
                widgets.HBox(
                    (widgets.HTML(format_html("Datasets:")), self._datasets_dropdown,)
                ),
                widgets.HBox(
                    (
                        widgets.HTML(format_html("Recalibrated Only:")),
                        self._recal_checkbox,
                    )
                ),
            )
        )
        self.filtered_callback = None

    def _init_domains(self):
        domains = [mod.domain_display for mod in self._all_models]
        domains = list(dict.fromkeys(domains))
        domains.sort()
        domains.insert(0, "all domains")

        return domains

    def _init_datasets(self):
        datasets = [mod.dataset for mod in self._all_models]
        datasets = list(dict.fromkeys(datasets))
        datasets.sort()
        datasets.insert(0, "all datasets")

        return datasets

    def _setup_hooks(self):
        def _invoke_callback():
            if (
                self._domains_dropdown.value
                and self._domains_dropdown.value != "all domains"
            ):
                domains = [self._domains_dropdown.value.split(" ")[0]]
                sub_domains = [self._domains_dropdown.value.split(" ")[1]]
            else:
                domains = None
                sub_domains = None

            datasets = (
                [self._datasets_dropdown.value]
                if self._datasets_dropdown.value != "all datasets"
                and self._datasets_dropdown.value
                else None
            )
            descs = ["recal", "recal-perf"] if self._recal_checkbox.value else None

            if self.filtered_callback:
                self.filtered_callback(domains, sub_domains, datasets, descs)

        def _recal_change(change):
            _invoke_callback()

        def _domains_change(change):
            self._selected_domain = change["new"]
            _invoke_callback()

        def _datasets_change(change):
            self._selected_dataset = change["new"]
            _invoke_callback()

        self._recal_checkbox.observe(_recal_change, names="value")
        self._domains_dropdown.observe(_domains_change, names="value")
        self._datasets_dropdown.observe(_datasets_change, names="value")


class _ModelsWidget(object):
    def __init__(self, forced_frameworks: Union[None, List[str]]):
        self._forced_frameworks = forced_frameworks
        self._architecture_selector = widgets.Select(
            options=[], description="Networks:"
        )
        self._dataset_selector = widgets.Select(options=[], description="Dataset:")
        self._framework_selector = (
            widgets.Select(options=[], description="ML Framework:")
            if not self._forced_frameworks or len(self._forced_frameworks) > 1
            else widgets.Box()
        )
        self._desc_selector = widgets.Select(options=[], description="Type:")
        self._selected_text = widgets.HTML(format_html("", italic=True))
        self.container = widgets.VBox(
            (
                widgets.HTML(format_html("Selection", header="h4")),
                widgets.HBox(
                    (
                        self._architecture_selector,
                        self._dataset_selector,
                        self._framework_selector,
                        self._desc_selector,
                    )
                ),
                widgets.HBox(
                    (
                        widgets.HTML(format_html("Selected:", header="h5")),
                        self._selected_text,
                    )
                ),
            )
        )
        self._filtered = []  # type: List[RepoModel]
        self._setup_hooks()
        self.selected = None

    @property
    def selected_framework(self) -> str:
        return self._framework_selector.value

    def update(self, filtered: List[RepoModel]):
        self._filtered = filtered
        self._update_selectors()

    def _setup_hooks(self):
        def _selector_change(change):
            if change["new"] != change["old"]:
                self._update_selectors()

        self._architecture_selector.observe(_selector_change, names="value")
        self._dataset_selector.observe(_selector_change, names="value")
        self._framework_selector.observe(_selector_change, names="value")
        self._desc_selector.observe(_selector_change, names="value")

    def _update_selectors(self):
        architecture = self._architecture_selector.value
        dataset = self._dataset_selector.value
        framework = (
            self._framework_selector.value
            if not self._forced_frameworks or len(self._forced_frameworks) > 1
            else self._forced_frameworks[0]
        )
        desc = self._desc_selector.value

        architectures = {mod.architecture_display for mod in self._filtered}
        architectures = list(architectures)
        architectures.sort()
        if architecture not in architectures:
            architecture = architectures[0] if len(architectures) > 0 else None
        self._architecture_selector.options = architectures
        self._architecture_selector.value = architecture

        datasets = {
            mod.dataset
            for mod in self._filtered
            if mod.architecture_display == architecture
        }
        datasets = list(datasets)
        datasets.sort()
        if dataset not in datasets:
            dataset = datasets[0] if len(datasets) > 0 else None
        self._dataset_selector.options = datasets
        self._dataset_selector.value = dataset

        if self._forced_frameworks is None:
            frameworks = {
                mod.framework
                for mod in self._filtered
                if mod.architecture_display == architecture and mod.dataset == dataset
            }
            frameworks.add("onnx")
            frameworks = list(frameworks)
            frameworks.sort()
            if framework not in frameworks:
                framework = frameworks[0] if len(frameworks) > 0 else None
            self._framework_selector.options = frameworks
            self._framework_selector.value = framework

        descs = {
            mod.desc
            for mod in self._filtered
            if mod.architecture_display == architecture
            and mod.dataset == dataset
            and (mod.framework == framework or framework == "onnx")
        }
        descs = list(descs)
        descs.sort()
        if desc not in descs:
            desc = descs[0] if len(descs) > 0 else None
        self._desc_selector.options = descs
        self._desc_selector.value = desc

        self._update_selected()

    def _update_selected(self):
        self.selected = None
        self._selected_text.value = ""

        for mod in self._filtered:
            matches_arch = mod.arch_display == self._architecture_selector.value
            matches_dataset = mod.dataset == self._dataset_selector.value
            matches_framework = (
                (self._forced_frameworks and mod.framework in self._forced_frameworks)
                or not self._forced_frameworks
                and (
                    mod.framework == self._framework_selector.value
                    or self._framework_selector.value == "onnx"
                )
            )
            matches_desc = mod.desc == self._desc_selector.value

            if matches_arch and matches_dataset and matches_framework and matches_desc:
                self.selected = mod
                self._selected_text.value = format_html(mod.summary, italic=True)
                break


class ModelSelectWidgetContainer(object):
    """
    Widget used in model repo notebooks for selecting a model for download

    :param filter_frameworks: if provided, will force all models
        to be one of these frameworks
    :param filter_datasets: if provided, will force all models
        to be trained on one of these datasets
    """

    def __init__(
        self, filter_frameworks: List[str] = None, filter_datasets: List[str] = None
    ):
        self._models = available_models(
            frameworks=filter_frameworks if filter_frameworks else None,
            datasets=filter_datasets if filter_datasets else None,
        )
        self._models_widget = _ModelsWidget(filter_frameworks)
        self._filter_widget = _FilterWidget(self._models)

    @property
    def selected_model(self) -> RepoModel:
        """
        :return: the selected model in the widget
        """
        return self._models_widget.selected

    @property
    def selected_framework(self) -> str:
        """
        :return: the selected framework in the widget
        """
        return self._models_widget.selected_framework

    def create(self):
        """
        :return: a created ipywidget that allows selection of models
        """

        def _filter_change_callback(
            domains: List[str],
            sub_domains: List[str],
            datasets: List[str],
            descs: List[str],
        ):
            filtered = []

            for mod in self._models:
                if not filter_model(
                    mod,
                    domains=domains,
                    sub_domains=sub_domains,
                    architectures=None,
                    sub_architectures=None,
                    datasets=datasets,
                    frameworks=None,
                    descs=descs,
                ):
                    filtered.append(mod)

            self._models_widget.update(filtered)

        self._filter_widget.filtered_callback = _filter_change_callback
        self._models_widget.update(self._models)

        return widgets.VBox(
            (
                widgets.HTML(format_html("Model Selector", header="h2")),
                self._filter_widget.container,
                self._models_widget.container,
            )
        )
