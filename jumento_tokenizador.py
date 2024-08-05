import os
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from typing import (
    AbstractSet,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

# Máximo de caracteres para codificar sem causar pyo3_runtime.PanicException
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# Comprimento máximo de caracteres consecutivos não-brancos ou brancos para divisão
MAX_NO_WHITESPACES_CHARS = 25_000

class Tokenizer:
    """
    Classe Tokenizer para codificação e decodificação de texto usando o tokenizador Tiktoken.
    """

    def __init__(self, model_path: str):
        """
        Inicializa o Tokenizer com um caminho de modelo.

        Args:
            model_path (str): Caminho para o arquivo de modelo Tiktoken.
        """
        assert os.path.isfile(model_path), f"O caminho do modelo {model_path} não existe."
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        
        # Definir tokens especiais
        special_tokens = [
            "", "", "", "", "", "", "", "", "", "",  # Tokens especiais
        ]
        reserved_tokens = [
            f"<|reserved_special_token_{i}|>"
            for i in range(256 - len(special_tokens))  # Ajustar a contagem de tokens para se adequar ao modelo
        ]
        special_tokens += reserved_tokens

        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self._get_pattern_string(),
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words = num_base_tokens + len(special_tokens)
        self.bos_id = self.special_tokens[""]
        self.eos_id = self.special_tokens[""]
        self.eot_id = self.special_tokens[""]
        self.eom_id = self.special_tokens[""]
        self.python_tag_id = self.special_tokens[""]
        self.pad_id = self.special_tokens[""]
        self.stop_tokens = [self.special_tokens[""], self.special_tokens[""]]

    @staticmethod
    def _get_pattern_string() -> str:
        """
        Retorna a string do padrão de expressão regular para tokenização de texto.

        Returns:
            str: Padrão de regex para tokenização.
        """
        return r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def encode(
        self,
        s: str,
        *,
        bos: bool = False,
        eos: bool = False,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Codifica uma string em uma lista de IDs de tokens.

        Args:
            s (str): A string de entrada a ser codificada.
            bos (bool): Se deve adicionar o token de início de sequência.
            eos (bool): Se deve adicionar o token de fim de sequência.
            allowed_special (Optional[Union[Literal["all"], AbstractSet[str]]]): Tokens especiais permitidos.
            disallowed_special (Union[Literal["all"], Collection[str]]): Tokens especiais não permitidos.

        Returns:
            List[int]: Uma lista de IDs de tokens.
        """
        if allowed_special is None:
            allowed_special = set()

        # Dividir a string para gerenciar os limites de codificação
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i:i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        tokens = []
        for substr in substrs:
            tokens.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            tokens.insert(0, self.bos_id)
        if eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        """
        Decodifica uma lista de IDs de tokens de volta para uma string.

        Args:
            token_ids (Sequence[int]): A lista de IDs de tokens a ser decodificada.

        Returns:
            str: A string decodificada.
        """
        return self.model.decode(list(token_ids))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Divide a string em pedaços para garantir que não haja mais de `max_consecutive_slice_len`
        caracteres consecutivos brancos ou não-brancos.

        Args:
            s (str): A string a ser dividida.
            max_consecutive_slice_len (int): Comprimento máximo de caracteres consecutivos.

        Yields:
            Iterator[str]: Pedaços da string original.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if s else False
        slice_start = 0

        for i, char in enumerate(s):
            is_now_space = char.isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]
