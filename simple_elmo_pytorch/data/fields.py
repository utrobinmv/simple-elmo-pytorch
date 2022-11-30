import torch
from collections import defaultdict

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union, TypeVar, Generic, Iterator

from simple_elmo_pytorch.data.vocabulary import Vocabulary

from simple_elmo_pytorch.data.token_indexers import TokenIndexer

from simple_elmo_pytorch.data.tokenizers import Token

from simple_elmo_pytorch.nn import util

TextFieldTensors = Dict[str, Dict[str, torch.Tensor]]

DataArray = TypeVar(
    "DataArray", torch.Tensor, Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]
)

class Field(Generic[DataArray]):
    """
    A `Field` is some piece of a data instance that ends up as an tensor in a model (either as an
    input or an output).  Data instances are just collections of fields.

    Fields go through up to two steps of processing: (1) tokenized fields are converted into token
    ids, (2) fields containing token ids (or any other numeric data) are padded (if necessary) and
    converted into tensors.  The `Field` API has methods around both of these steps, though they
    may not be needed for some concrete `Field` classes - if your field doesn't have any strings
    that need indexing, you don't need to implement `count_vocab_items` or `index`.  These
    methods `pass` by default.

    Once a vocabulary is computed and all fields are indexed, we will determine padding lengths,
    then intelligently batch together instances and pad them into actual tensors.
    """

    __slots__ = []  # type: ignore

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        If there are strings in this field that need to be converted into integers through a
        :class:`Vocabulary`, here is where we count them, to determine which tokens are in or out
        of the vocabulary.

        If your `Field` does not have any strings that need to be converted into indices, you do
        not need to implement this method.

        A note on this `counter`: because `Fields` can represent conceptually different things,
        we separate the vocabulary items by `namespaces`.  This way, we can use a single shared
        mechanism to handle all mappings from strings to integers in all fields, while keeping
        words in a `TextField` from sharing the same ids with labels in a `LabelField` (e.g.,
        "entailment" or "contradiction" are labels in an entailment task)

        Additionally, a single `Field` might want to use multiple namespaces - `TextFields` can
        be represented as a combination of word ids and character ids, and you don't want words and
        characters to share the same vocabulary - "a" as a word should get a different id from "a"
        as a character, and the vocabulary sizes of words and characters are very different.

        Because of this, the first key in the `counter` object is a `namespace`, like "tokens",
        "token_characters", "tags", or "labels", and the second key is the actual vocabulary item.
        """
        pass

    def human_readable_repr(self) -> Any:
        """
        This method should be implemented by subclasses to return a structured, yet human-readable
        representation of the field.

        !!! Note
            `human_readable_repr()` is not meant to be used as a method to serialize a `Field` since the return
            value does not necessarily contain all of the attributes of the `Field` instance. But the object
            returned should be JSON-serializable.
        """
        raise NotImplementedError

    def index(self, vocab: Vocabulary):
        """
        Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
        This `modifies` the `Field` object, it does not return anything.

        If your `Field` does not have any strings that need to be converted into indices, you do
        not need to implement this method.
        """
        pass

    def get_padding_lengths(self) -> Dict[str, int]:
        """
        If there are things in this field that need padding, note them here.  In order to pad a
        batch of instance, we get all of the lengths from the batch, take the max, and pad
        everything to that length (or use a pre-specified maximum length).  The return value is a
        dictionary mapping keys to lengths, like `{'num_tokens': 13}`.

        This is always called after :func:`index`.
        """
        raise NotImplementedError

    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        """
        Given a set of specified padding lengths, actually pad the data in this field and return a
        torch Tensor (or a more complex data structure) of the correct shape.  We also take a
        couple of parameters that are important when constructing torch Tensors.

        # Parameters

        padding_lengths : `Dict[str, int]`
            This dictionary will have the same keys that were produced in
            :func:`get_padding_lengths`.  The values specify the lengths to use when padding each
            relevant dimension, aggregated across all instances in a batch.
        """
        raise NotImplementedError

    def empty_field(self) -> "Field":
        """
        So that `ListField` can pad the number of fields in a list (e.g., the number of answer
        option `TextFields`), we need a representation of an empty field of each type.  This
        returns that.  This will only ever be called when we're to the point of calling
        :func:`as_tensor`, so you don't need to worry about `get_padding_lengths`,
        `count_vocab_items`, etc., being called on this empty field.

        We make this an instance method instead of a static method so that if there is any state
        in the Field, we can copy it over (e.g., the token indexers in `TextField`).
        """
        raise NotImplementedError

    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:  # type: ignore
        """
        Takes the output of `Field.as_tensor()` from a list of `Instances` and merges it into
        one batched tensor for this `Field`.  The default implementation here in the base class
        handles cases where `as_tensor` returns a single torch tensor per instance.  If your
        subclass returns something other than this, you need to override this method.

        This operation does not modify `self`, but in some cases we need the information
        contained in `self` in order to perform the batching, so this is an instance method, not
        a class method.
        """

        return torch.stack(tensor_list)

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            # With the way "slots" classes work, self.__slots__ only gives the slots defined
            # by the current class, but not any of its base classes. Therefore to truly
            # check for equality we have to check through all of the slots in all of the
            # base classes as well.
            for class_ in self.__class__.mro():
                for attr in getattr(class_, "__slots__", []):
                    if getattr(self, attr) != getattr(other, attr):
                        return False
            # It's possible that a subclass was not defined as a slots class, in which
            # case we'll need to check __dict__.
            if hasattr(self, "__dict__"):
                return self.__dict__ == other.__dict__
            return True
        return NotImplemented

    def __len__(self):
        raise NotImplementedError

    def duplicate(self):
        return deepcopy(self)

class SequenceField(Field[DataArray]):
    """
    A `SequenceField` represents a sequence of things.  This class just adds a method onto
    `Field`: :func:`sequence_length`.  It exists so that `SequenceLabelField`, `IndexField` and other
    similar `Fields` can have a single type to require, with a consistent API, whether they are
    pointing to words in a `TextField`, items in a `ListField`, or something else.
    """

    __slots__ = []  # type: ignore

    def sequence_length(self) -> int:
        """
        How many elements are there in this sequence?
        """
        raise NotImplementedError

    def empty_field(self) -> "SequenceField":
        raise NotImplementedError

class TextField(SequenceField[TextFieldTensors]):
    """
    This `Field` represents a list of string tokens.  Before constructing this object, you need
    to tokenize raw strings using a :class:`~allennlp.data.tokenizers.tokenizer.Tokenizer`.

    Because string tokens can be represented as indexed arrays in a number of ways, we also take a
    dictionary of :class:`~allennlp.data.token_indexers.token_indexer.TokenIndexer`
    objects that will be used to convert the tokens into indices.
    Each `TokenIndexer` could represent each token as a single ID, or a list of character IDs, or
    something else.

    This field will get converted into a dictionary of arrays, one for each `TokenIndexer`.  A
    `SingleIdTokenIndexer` produces an array of shape (num_tokens,), while a
    `TokenCharactersIndexer` produces an array of shape (num_tokens, num_characters).
    """

    __slots__ = ["tokens", "_token_indexers", "_indexed_tokens"]

    def __init__(
        self, tokens: List[Token], token_indexers: Optional[Dict[str, TokenIndexer]] = None
    ) -> None:
        self.tokens = tokens
        self._token_indexers = token_indexers
        self._indexed_tokens: Optional[Dict[str, IndexedTokenList]] = None

        if not all(isinstance(x, (Token)) for x in tokens):
            raise ConfigurationError(
                "TextFields must be passed Tokens. "
                "Found: {} with types {}.".format(tokens, [type(x) for x in tokens])
            )

    @property
    def token_indexers(self) -> Dict[str, TokenIndexer]:
        if self._token_indexers is None:
            raise ValueError(
                "TextField's token_indexers have not been set.\n"
                "Did you forget to call DatasetReader.apply_token_indexers(instance) "
                "on your instance?\n"
                "If apply_token_indexers() is being called but "
                "you're still seeing this error, it may not be implemented correctly."
            )
        return self._token_indexers

    @token_indexers.setter
    def token_indexers(self, token_indexers: Dict[str, TokenIndexer]) -> None:
        self._token_indexers = token_indexers

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self.token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)

    def index(self, vocab: Vocabulary):
        self._indexed_tokens = {}
        for indexer_name, indexer in self.token_indexers.items():
            self._indexed_tokens[indexer_name] = indexer.tokens_to_indices(self.tokens, vocab)

    def get_padding_lengths(self) -> Dict[str, int]:
        """
        The `TextField` has a list of `Tokens`, and each `Token` gets converted into arrays by
        (potentially) several `TokenIndexers`.  This method gets the max length (over tokens)
        associated with each of these arrays.
        """
        if self._indexed_tokens is None:
            raise ConfigurationError(
                "You must call .index(vocabulary) on a field before determining padding lengths."
            )

        padding_lengths = {}
        for indexer_name, indexer in self.token_indexers.items():
            indexer_lengths = indexer.get_padding_lengths(self._indexed_tokens[indexer_name])
            for key, length in indexer_lengths.items():
                padding_lengths[f"{indexer_name}___{key}"] = length
        return padding_lengths

    def sequence_length(self) -> int:
        return len(self.tokens)

    def as_tensor(self, padding_lengths: Dict[str, int]) -> TextFieldTensors:
        if self._indexed_tokens is None:
            raise ConfigurationError(
                "You must call .index(vocabulary) on a field before calling .as_tensor()"
            )

        tensors = {}

        indexer_lengths: Dict[str, Dict[str, int]] = defaultdict(dict)
        for key, value in padding_lengths.items():
            # We want this to crash if the split fails. Should never happen, so I'm not
            # putting in a check, but if you fail on this line, open a github issue.
            indexer_name, padding_key = key.split("___")
            indexer_lengths[indexer_name][padding_key] = value

        for indexer_name, indexer in self.token_indexers.items():
            tensors[indexer_name] = indexer.as_padded_tensor_dict(
                self._indexed_tokens[indexer_name], indexer_lengths[indexer_name]
            )
        return tensors

    def empty_field(self):
        text_field = TextField([], self._token_indexers)
        text_field._indexed_tokens = {}
        if self._token_indexers is not None:
            for indexer_name, indexer in self.token_indexers.items():
                text_field._indexed_tokens[indexer_name] = indexer.get_empty_token_list()
        return text_field

    def batch_tensors(self, tensor_list: List[TextFieldTensors]) -> TextFieldTensors:
        # This is creating a dict of {token_indexer_name: {token_indexer_outputs: batched_tensor}}
        # for each token indexer used to index this field.
        indexer_lists: Dict[str, List[Dict[str, torch.Tensor]]] = defaultdict(list)
        for tensor_dict in tensor_list:
            for indexer_name, indexer_output in tensor_dict.items():
                indexer_lists[indexer_name].append(indexer_output)
        batched_tensors = {
            # NOTE(mattg): if an indexer has its own nested structure, rather than one tensor per
            # argument, then this will break.  If that ever happens, we should move this to an
            # `indexer.batch_tensors` method, with this logic as the default implementation in the
            # base class.
            indexer_name: util.batch_tensor_dicts(indexer_outputs)
            for indexer_name, indexer_outputs in indexer_lists.items()
        }
        return batched_tensors

    def __str__(self) -> str:
        # Double tab to indent under the header.
        formatted_text = "".join(
            "\t\t" + text + "\n" for text in textwrap.wrap(repr(self.tokens), 100)
        )
        if self._token_indexers is not None:
            indexers = {
                name: indexer.__class__.__name__ for name, indexer in self._token_indexers.items()
            }
            return (
                f"TextField of length {self.sequence_length()} with "
                f"text: \n {formatted_text} \t\tand TokenIndexers : {indexers}"
            )
        else:
            return f"TextField of length {self.sequence_length()} with text: \n {formatted_text}"

    # Sequence[Token] methods
    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    def duplicate(self):
        """
        Overrides the behavior of `duplicate` so that `self._token_indexers` won't
        actually be deep-copied.

        Not only would it be extremely inefficient to deep-copy the token indexers,
        but it also fails in many cases since some tokenizers (like those used in
        the 'transformers' lib) cannot actually be deep-copied.
        """
        if self._token_indexers is not None:
            new = TextField(deepcopy(self.tokens), {k: v for k, v in self._token_indexers.items()})
        else:
            new = TextField(deepcopy(self.tokens))
        new._indexed_tokens = deepcopy(self._indexed_tokens)
        return new

    def human_readable_repr(self) -> List[str]:
        return [str(t) for t in self.tokens]