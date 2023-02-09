import re
import unicodedata
from typing import List
from typing import Optional

from jptranstokenizer.mainword.base import MainTokenizerABC


class WinJumanTokenizer(MainTokenizerABC):
    """Tokenizer to split into words using Juman.
    Juman++ and pyknp are required to use.
    You can import this module shortly:

    .. code-block:: none

       >> from jptranstokenizer.mainword import JumanTokenizer

    Args:
        do_lower_case (``bool``, *optional*, defaults to ``False``):
            Whether or not to lowercase the input when tokenizing.Defaults to None.
        normalize_text (``bool``, *optional*, defaults to ``True``):
            Whether to apply unicode normalization to text before tokenization.
        ignore_max_byte_error (``bool``, *optional*, defaults to ``False``):
            Whether or not to ignore error of max bytes (only valid with Juman and Sudachi).
            If valid, the tokenizer return empty list.

    .. seealso::
        - Juman++ https://github.com/ku-nlp/jumanpp
        - pyknp https://github.com/ku-nlp/pyknp
    """

    def __init__(
        self,
        executable_path: Optional[str] = None,
        do_lower_case: bool = False,
        normalize_text: bool = True,
        ignore_max_byte_error: bool = False,
    ):
        super().__init__(do_lower_case=do_lower_case, normalize_text=normalize_text)
        self.ignore_max_byte_error = ignore_max_byte_error
        try:
            from rhoknp import Jumanpp
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install rhoknp to use JumanTokenizer."
                "See https://github.com/ku-nlp/rhoknp for installation."
            )
        if executable_path is None:
            self.juman = Jumanpp()
        else:
            self.juman = Jumanpp(executable_path)

    def tokenize(self, text, never_split=None, **kwargs) -> List[str]:
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)
        # "#" and "@" at the beginning of a sentence causes timeout error
        text = re.sub("^#", "＃", text)
        text = re.sub("^@", "＠", text)
        tokens: List[str] = []
        if not self.ignore_max_byte_error or len(text.encode()) <= 4096:
            try:
                result = self.juman.apply_to_sentence(text).morphemes
                use_underscore = False
                use_quote = False
            except ValueError:
                # This error is occured because of the Juman's matter about space
                if '"' in text:
                    text = text.replace('"', "”")
                    use_quote = True
                else:
                    use_quote = False
                if re.search(r"\s", text):
                    text = re.sub(r"\s", "_", text)
                    use_underscore = True
                else:
                    use_underscore = False
                try:
                    result = self.juman.analysis(text)
                except Exception:
                    print(text)
                    import sys

                    sys.exit(1)
            except Exception:
                print(text)
                import sys

                sys.exit(1)
            for mrph in result:
                token = mrph.text
                if self.do_lower_case:
                    token = token.lower()
                tokens.append(token)
            if use_underscore:
                tokens = list(filter(lambda x: x != "_", tokens))
            if use_quote:
                tokens = list(map(lambda x: x.replace("”", '"'), tokens))
        return tokens
