import torch
import logging

from torch.nn.functional import embedding

from simple_elmo_pytorch.checks import ConfigurationError
from simple_elmo_pytorch.data.vocabulary import Vocabulary

from simple_elmo_pytorch.nn import util

from simple_elmo_pytorch.file_utils import get_file_extension

logger = logging.getLogger(__name__)

class TokenEmbedder(torch.nn.Module): #, Registrable
    """
    A `TokenEmbedder` is a `Module` that takes as input a tensor with integer ids that have
    been output from a [`TokenIndexer`](/api/data/token_indexers/token_indexer.md) and outputs
    a vector per token in the input.  The input typically has shape `(batch_size, num_tokens)`
    or `(batch_size, num_tokens, num_characters)`, and the output is of shape `(batch_size, num_tokens,
    output_dim)`.  The simplest `TokenEmbedder` is just an embedding layer, but for
    character-level input, it could also be some kind of character encoder.

    We add a single method to the basic `Module` API: `get_output_dim()`.  This lets us
    more easily compute output dimensions for the
    [`TextFieldEmbedder`](/api/modules/text_field_embedders/text_field_embedder.md),
    which we might need when defining model parameters such as LSTMs or linear layers, which need
    to know their input dimension before the layers are called.
    """

    default_implementation = "embedding"

    def get_output_dim(self) -> int:
        """
        Returns the final output dimension that this `TokenEmbedder` uses to represent each
        token.  This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

class Embedding(TokenEmbedder):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:

        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only makes sense with
           non-trainable embeddings).

    Note that if you are using our data API and are trying to embed a
    [`TextField`](../../data/fields/text_field.md), you should use a
    [`TextFieldEmbedder`](../text_field_embedders/text_field_embedder.md) instead of using this directly.

    Registered as a `TokenEmbedder` with name "embedding".

    # Parameters

    num_embeddings : `int`
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : `int`
        The size of each embedding vector.
    projection_dim : `int`, optional (default=`None`)
        If given, we add a projection layer after the embedding layer.  This really only makes
        sense if `trainable` is `False`.
    weight : `torch.FloatTensor`, optional (default=`None`)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index : `int`, optional (default=`None`)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : `bool`, optional (default=`True`)
        Whether or not to optimize the embedding parameters.
    max_norm : `float`, optional (default=`None`)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type : `float`, optional (default=`2`)
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : `bool`, optional (default=`False`)
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : `bool`, optional (default=`False`)
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.
    vocab_namespace : `str`, optional (default=`None`)
        In case of fine-tuning/transfer learning, the model's embedding matrix needs to be
        extended according to the size of extended-vocabulary. To be able to know how much to
        extend the embedding-matrix, it's necessary to know which vocab_namspace was used to
        construct it in the original training. We store vocab_namespace used during the original
        training as an attribute, so that it can be retrieved during fine-tuning.
    pretrained_file : `str`, optional (default=`None`)
        Path to a file of word vectors to initialize the embedding matrix. It can be the
        path to a local file or a URL of a (cached) remote file. Two formats are supported:
            * hdf5 file - containing an embedding matrix in the form of a torch.Tensor;
            * text file - an utf-8 encoded text file with space separated fields.
    vocab : `Vocabulary`, optional (default = `None`)
        Used to construct an embedding from a pretrained file.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "embedding", it gets specified as a top-level parameter, then is passed in to this module
        separately.

    # Returns

    An Embedding module.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = None,
        projection_dim: int = None,
        weight: torch.FloatTensor = None,
        padding_index: int = None,
        trainable: bool = True,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        vocab_namespace: str = "tokens",
        pretrained_file: str = None,
        vocab: Vocabulary = None,
    ) -> None:
        super().__init__()

        if num_embeddings is None and vocab is None:
            raise ConfigurationError(
                "Embedding must be constructed with either num_embeddings or a vocabulary."
            )

        _vocab_namespace: Optional[str] = vocab_namespace
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(_vocab_namespace)  # type: ignore
        else:
            # If num_embeddings is present, set default namespace to None so that extend_vocab
            # call doesn't misinterpret that some namespace was originally used.
            _vocab_namespace = None  # type: ignore

        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = _vocab_namespace
        self._pretrained_file = pretrained_file

        self.output_dim = projection_dim or embedding_dim

        if weight is not None and pretrained_file:
            raise ConfigurationError(
                "Embedding was constructed with both a weight and a pretrained file."
            )

        elif pretrained_file is not None:

            if vocab is None:
                raise ConfigurationError(
                    "To construct an Embedding from a pretrained file, you must also pass a vocabulary."
                )

            # If we're loading a saved model, we don't want to actually read a pre-trained
            # embedding file - the embeddings will just be in our saved weights, and we might not
            # have the original embedding file anymore, anyway.

            # TODO: having to pass tokens here is SUPER gross, but otherwise this breaks the
            # extend_vocab method, which relies on the value of vocab_namespace being None
            # to infer at what stage the embedding has been constructed. Phew.
            weight = _read_pretrained_embeddings_file(
                pretrained_file, embedding_dim, vocab, vocab_namespace
            )
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        elif weight is not None:
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        else:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)

        # Whatever way we have constructed the embedding, it should be consistent with
        # num_embeddings and embedding_dim.
        if self.weight.size() != (num_embeddings, embedding_dim):
            raise ConfigurationError(
                "A weight matrix was passed with contradictory embedding shapes."
            )

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass tokens to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = tokens.size()
        tokens = util.combine_initial_dims(tokens)

        embedded = embedding(
            tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

    def extend_vocab(
        self,
        extended_vocab: Vocabulary,
        vocab_namespace: str = None,
        extension_pretrained_file: str = None,
        model_path: str = None,
    ):
        """
        Extends the embedding matrix according to the extended vocabulary.
        If extension_pretrained_file is available, it will be used for initializing the new words
        embeddings in the extended vocabulary; otherwise we will check if _pretrained_file attribute
        is already available. If none is available, they will be initialized with xavier uniform.

        # Parameters

        extended_vocab : `Vocabulary`
            Vocabulary extended from original vocabulary used to construct
            this `Embedding`.
        vocab_namespace : `str`, (optional, default=`None`)
            In case you know what vocab_namespace should be used for extension, you
            can pass it. If not passed, it will check if vocab_namespace used at the
            time of `Embedding` construction is available. If so, this namespace
            will be used or else extend_vocab will be a no-op.
        extension_pretrained_file : `str`, (optional, default=`None`)
            A file containing pretrained embeddings can be specified here. It can be
            the path to a local file or an URL of a (cached) remote file. Check format
            details in `from_params` of `Embedding` class.
        model_path : `str`, (optional, default=`None`)
            Path traversing the model attributes upto this embedding module.
            Eg. "_text_field_embedder.token_embedder_tokens". This is only useful
            to give a helpful error message when extend_vocab is implicitly called
            by train or any other command.
        """
        # Caveat: For allennlp v0.8.1 and below, we weren't storing vocab_namespace as an attribute,
        # knowing which is necessary at time of embedding vocab extension. So old archive models are
        # currently unextendable.

        vocab_namespace = vocab_namespace or self._vocab_namespace
        if not vocab_namespace:
            # It's not safe to default to "tokens" or any other namespace.
            logger.info(
                "Loading a model trained before embedding extension was implemented; "
                "pass an explicit vocab namespace if you want to extend the vocabulary."
            )
            return

        extended_num_embeddings = extended_vocab.get_vocab_size(vocab_namespace)
        if extended_num_embeddings == self.num_embeddings:
            # It's already been extended. No need to initialize / read pretrained file in first place (no-op)
            return

        if extended_num_embeddings < self.num_embeddings:
            raise ConfigurationError(
                f"Size of namespace, {vocab_namespace} for extended_vocab is smaller than "
                f"embedding. You likely passed incorrect vocab or namespace for extension."
            )

        # Case 1: user passed extension_pretrained_file and it's available.
        if extension_pretrained_file and is_url_or_existing_file(extension_pretrained_file):
            # Don't have to do anything here, this is the happy case.
            pass
        # Case 2: user passed extension_pretrained_file and it's not available
        elif extension_pretrained_file:
            raise ConfigurationError(
                f"You passed pretrained embedding file {extension_pretrained_file} "
                f"for model_path {model_path} but it's not available."
            )
        # Case 3: user didn't pass extension_pretrained_file, but pretrained_file attribute was
        # saved during training and is available.
        elif is_url_or_existing_file(self._pretrained_file):
            extension_pretrained_file = self._pretrained_file
        # Case 4: no file is available, hope that pretrained embeddings weren't used in the first place and warn
        elif self._pretrained_file is not None:
            # Warn here instead of an exception to allow a fine-tuning even without the original pretrained_file
            logger.warning(
                f"Embedding at model_path, {model_path} cannot locate the pretrained_file. "
                f"Originally pretrained_file was at '{self._pretrained_file}'."
            )
        else:
            # When loading a model from archive there is no way to distinguish between whether a pretrained-file
            # was or wasn't used during the original training. So we leave an info here.
            logger.info(
                "If you are fine-tuning and want to use a pretrained_file for "
                "embedding extension, please pass the mapping by --embedding-sources argument."
            )

        embedding_dim = self.weight.data.shape[-1]
        if not extension_pretrained_file:
            extra_num_embeddings = extended_num_embeddings - self.num_embeddings
            extra_weight = torch.FloatTensor(extra_num_embeddings, embedding_dim)
            torch.nn.init.xavier_uniform_(extra_weight)
        else:
            # It's easiest to just reload the embeddings for the entire vocab,
            # then only keep the ones we need.
            whole_weight = _read_pretrained_embeddings_file(
                extension_pretrained_file, embedding_dim, extended_vocab, vocab_namespace
            )
            extra_weight = whole_weight[self.num_embeddings :, :]

        device = self.weight.data.device
        extended_weight = torch.cat([self.weight.data, extra_weight.to(device)], dim=0)
        self.weight = torch.nn.Parameter(extended_weight, requires_grad=self.weight.requires_grad)
        self.num_embeddings = extended_num_embeddings



def _read_pretrained_embeddings_file(
    file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
) -> torch.FloatTensor:
    """
    Returns and embedding matrix for the given vocabulary using the pretrained embeddings
    contained in the given file. Embeddings for tokens not found in the pretrained embedding file
    are randomly initialized using a normal distribution with mean and standard deviation equal to
    those of the pretrained embeddings.

    We support two file formats:

        * text format - utf-8 encoded text file with space separated fields: [word] [dim 1] [dim 2] ...
          The text file can eventually be compressed, and even resides in an archive with multiple files.
          If the file resides in an archive with other files, then `embeddings_filename` must
          be a URI "(archive_uri)#file_path_inside_the_archive"

        * hdf5 format - hdf5 file containing an embedding matrix in the form of a torch.Tensor.

    If the filename ends with '.hdf5' or '.h5' then we load from hdf5, otherwise we assume
    text format.

    # Parameters

    file_uri : `str`, required.
        It can be:

        * a file system path or a URL of an eventually compressed text file or a zip/tar archive
          containing a single file.

        * URI of the type `(archive_path_or_url)#file_path_inside_archive` if the text file
          is contained in a multi-file archive.

    vocab : `Vocabulary`, required.
        A Vocabulary object.
    namespace : `str`, (optional, default=`"tokens"`)
        The namespace of the vocabulary to find pretrained embeddings for.
    trainable : `bool`, (optional, default=`True`)
        Whether or not the embedding parameters should be optimized.

    # Returns

    A weight matrix with embeddings initialized from the read file.  The matrix has shape
    `(vocab.get_vocab_size(namespace), embedding_dim)`, where the indices of words appearing in
    the pretrained embedding file are initialized to the pretrained embedding value.
    """
    file_ext = get_file_extension(file_uri)
    if file_ext in [".h5", ".hdf5"]:
        return _read_embeddings_from_hdf5(file_uri, embedding_dim, vocab, namespace)

    return _read_embeddings_from_text_file(file_uri, embedding_dim, vocab, namespace)
