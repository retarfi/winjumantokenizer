import os
from typing import Optional
from typing import Union

from jptranstokenizer import JapaneseTransformerTokenizer

from .juman import WinJumanTokenizer


def load_berttokenizer_for_winjuman(
    tokenizer_path: Union[str, os.PathLike],
    executable_path: Optional[str] = None,
) -> JapaneseTransformerTokenizer:
    tokenizer: JapaneseTransformerTokenizer = (
        JapaneseTransformerTokenizer.from_pretrained(
            tokenizer_name_or_path=tokenizer_path,
            word_tokenizer_type="basic",
            tokenizer_class="AlbertTokenizer",
        )
    )
    tokenizer.word_tokenizer = WinJumanTokenizer(executable_path)
    return tokenizer
