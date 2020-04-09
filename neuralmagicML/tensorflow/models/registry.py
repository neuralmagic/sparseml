"""
Code related to the PyTorch model registry for easily creating models.
"""

from typing import Union, List, Callable, Any, Dict

from neuralmagicML.utils import TENSORFLOW_FRAMEWORK
from neuralmagicML.utils import RepoModel
from neuralmagicML.tensorflow.utils import tf_compat


__all__ = ["ModelRegistry"]


class _ModelAttributes(object):
    def __init__(
        self,
        input_shape: Any,
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        default_dataset: str,
        default_desc: str,
    ):
        self.input_shape = input_shape
        self.domain = domain
        self.sub_domain = sub_domain
        self.architecture = architecture
        self.sub_architecture = sub_architecture
        self.default_dataset = default_dataset
        self.default_desc = default_desc


class ModelRegistry(object):
    """
    Registry class for creating models
    """

    _CONSTRUCTORS = {}  # type: Dict[str, Callable]
    _ATTRIBUTES = {}  # type: Dict[str, _ModelAttributes]

    @staticmethod
    def create(key: str, *args, **kwargs) -> Any:
        """
        Create a new model for the given key

        :param key: the model key (name) to create
        :param args: any args to supply to the graph constructor
        :param kwargs: any keyword args to supply to the graph constructor
        :return: the outputs from the created graph
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )

        return ModelRegistry._CONSTRUCTORS[key](*args, **kwargs)

    @staticmethod
    def load_pretrained(
        key: str,
        pretrained: Union[bool, str] = True,
        pretrained_dataset: str = None,
        pretrained_path: str = None,
        sess: tf_compat.Session = None,
        saver: tf_compat.train.Saver = None,
    ):
        """
        Load pre-trained variables for a given model into a session.
        Uses a Saver object from TensorFlow to restore the variables
        from an index and data file.

        :param key: the model key (name) to create
        :param pretrained: True to load the default pretrained variables,
            a string to load a specific pretrained graph
            (ex: dense, recal, recal-perf),
            or False to not load any pretrained weights
        :param pretrained_dataset: The dataset to load pretrained weights for
            (ex: imagenet, mnist, etc).
            If not supplied will default to the one preconfigured for the model.
        :param pretrained_path: A path to the pretrained variables to load,
            if provided will override the pretrained param
        :param sess: The session to load the model variables into
            if pretrained_path or pretrained is supplied.
            If not supplied and required, then will use the default session
        :param saver: The Saver instance to use to restore the variables
            for the graph if pretrained_path or pretrained is supplied.
            If not supplied and required, then will create one using the
            TRAINABLE_VARIABLES collection
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )

        attributes = ModelRegistry._ATTRIBUTES[key]

        if not sess and (pretrained_path or pretrained):
            sess = tf_compat.get_default_session()

        if not saver and (pretrained_path or pretrained):
            saver = tf_compat.train.Saver(
                tf_compat.get_collection(tf_compat.GraphKeys.TRAINABLE_VARIABLES)
            )

        if pretrained_path:
            saver.restore(sess, pretrained_path)
        elif pretrained:
            desc = (
                pretrained if isinstance(pretrained, str) else attributes.default_desc
            )
            dataset = (
                pretrained_dataset if pretrained_dataset else attributes.default_dataset
            )
            repo_model = RepoModel(
                attributes.domain,
                attributes.sub_domain,
                attributes.architecture,
                attributes.sub_architecture,
                dataset,
                TENSORFLOW_FRAMEWORK,
                desc,
            )
            try:
                paths = repo_model.download_framework_files()
                index_path = [path for path in paths if path.endswith(".index")]
                index_path = index_path[0]
                model_path = index_path[:-6]
                saver.restore(sess, model_path)
            except Exception as ex:
                # try one more time with overwrite on in case files were corrupted
                paths = repo_model.download_framework_files(overwrite=True)
                index_path = [path for path in paths if path.endswith(".index")]

                if len(index_path) != 1:
                    raise FileNotFoundError(
                        "could not find .index file for {}".format(repo_model.root_path)
                    )

                index_path = index_path[0]
                model_path = index_path[:-6]
                saver.restore(sess, model_path)

    @staticmethod
    def input_shape(key: str):
        """
        :param key: the model key (name) to create
        :return: the specified input shape for the model
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )

        return ModelRegistry._ATTRIBUTES[key].input_shape

    @staticmethod
    def register(
        key: Union[str, List[str]],
        input_shape: Any,
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        default_dataset: str,
        default_desc: str,
    ):
        """
        Register a model with the registry. Should be used as a decorator

        :param key: the model key (name) to create
        :param input_shape: the specified input shape for the model
        :param domain: the domain the model belongs to; ex: cv, nlp, etc
        :param sub_domain: the sub domain the model belongs to;
            ex: classification, detection, etc
        :param architecture: the architecture the model belongs to;
            ex: resnet, mobilenet, etc
        :param sub_architecture: the sub architecture the model belongs to;
            ex: 50, 101, etc
        :param default_dataset: the dataset to use by default for loading
            pretrained if not supplied
        :param default_desc: the description to use by default for loading
            pretrained if not supplied
        :return: the decorator
        """
        if not isinstance(key, List):
            key = [key]

        def decorator(const_func):
            for r_key in key:
                if r_key in ModelRegistry._CONSTRUCTORS:
                    raise ValueError("key {} is already registered".format(key))

                ModelRegistry._CONSTRUCTORS[r_key] = const_func
                ModelRegistry._ATTRIBUTES[r_key] = _ModelAttributes(
                    input_shape,
                    domain,
                    sub_domain,
                    architecture,
                    sub_architecture,
                    default_dataset,
                    default_desc,
                )

            return const_func

        return decorator
