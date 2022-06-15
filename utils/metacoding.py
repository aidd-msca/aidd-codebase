import warnings
from abc import abstractclassmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


class CreditType(Enum):
    REFERENCE = auto()
    ACKNOWLEDGEMENT = auto()
    NONE = auto()


class Accreditation(Singleton):
    accreditations: Dict[str, Tuple[str, CreditType]] = {}
    called_creditations: Dict[str, Tuple[str, CreditType]] = {}

    def register_accreditation(
        self, call_name: str, accreditation: str, credit_type: CreditType
    ) -> None:
        self.accreditations[call_name] = (accreditation, credit_type)

    def get_accreditation(self, call_name: str) -> Tuple[str, CreditType]:
        return self.accreditations[call_name]

    def notice_accreditation(self, call_name: str) -> None:
        if call_name in self.called_creditations.keys():
            pass
        else:
            try:
                self.called_creditations[call_name] = self.get_accreditation(
                    call_name
                )
            except Exception:
                pass

    def choice_called(self, call_name: str) -> None:
        self.notice_accreditation(call_name)

    def return_accreditations(self) -> Dict[str, Tuple[str, CreditType]]:
        return self.called_creditations


class DictChoiceFactory:
    """Factory to generate a dictionary choice class"""

    dict_choice: Dict[str, Callable] = {}
    accreditation: Accreditation = Accreditation()

    @classmethod
    def register_choice(
        cls,
        call_name: str,
        accreditation: Optional[str],
        credit_type: Optional[CreditType],
    ):
        """Registers choices by string with a decorator.
        Optional: registers accreditation.
        """

        def wrapper(callable_cls: Any) -> Any:
            if accreditation and credit_type:
                cls.accreditation.register_accreditation(
                    call_name, accreditation, credit_type
                )
            cls.dict_choice[call_name] = callable_cls
            return callable_cls

        return wrapper

    @classmethod
    def register_prebuilt_choice(
        cls,
        call_name: str,
        callable_cls: Any,
        accreditation: Optional[str],
        credit_type: Optional[CreditType],
    ) -> Any:
        """Registers choices by string directly.
        Optional: registers accreditation.
        """
        if accreditation and credit_type:
            cls.accreditation.register_accreditation(
                call_name, accreditation, credit_type
            )
        cls.dict_choice[call_name] = callable_cls

    @abstractclassmethod
    def get_choice(cls, call_name: str) -> Callable:
        """Returns a choice pointer out of the dict_choice dictionary."""
        cls.accreditation.choice_called(call_name)
        return cls.dict_choice[call_name]

    @classmethod
    def check_choice(cls, call_name: str) -> bool:
        """Checks if a choice is valid and returns a bool."""
        if call_name not in cls.dict_choice.keys():
            warnings.warn(f"{call_name} is not recognized.")
            return False
        return True

    @classmethod
    def validate_choice(cls, call_name: str) -> None:
        """Checks if a choice is valid and stops if not."""
        if call_name not in cls.dict_choice.keys():
            raise ValueError(f"{call_name} is not recognized.")

    @classmethod
    def choice_accreditations(cls) -> Dict[str, Tuple[str, CreditType]]:
        return cls.accreditation.return_accreditations()
