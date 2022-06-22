import warnings
from abc import abstractclassmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple

from .config import _ABCDataClass


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


class Borg:
    _dict = None

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        if cls._dict is None:
            cls._dict = obj.__dict__
        else:
            obj.__dict__ = cls._dict
        return obj


class CreditType(Enum):
    REFERENCE = auto()
    ACKNOWLEDGEMENT = auto()
    NONE = auto()


class Accreditation(Singleton):
    accreditations: Dict[
        str,
        Tuple[Tuple[Optional[str], Optional[str], Optional[str]], CreditType],
    ] = {}
    called_creditations: Dict[
        str,
        Tuple[Tuple[Optional[str], Optional[str], Optional[str]], CreditType],
    ] = {}

    def register_accreditation(
        self,
        call_name: str,
        author: str = None,
        github_handle: str = None,
        additional_information: str = None,
        credit_type: CreditType = CreditType.NONE,
    ) -> None:
        self.accreditations[call_name] = (
            (author, github_handle, additional_information),
            credit_type,
        )

    def get_accreditation(
        self, call_name: str
    ) -> Tuple[Tuple[Optional[str], Optional[str], Optional[str]], CreditType]:
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

    def return_accreditations(
        self,
    ) -> Dict[
        str,
        Tuple[Tuple[Optional[str], Optional[str], Optional[str]], CreditType],
    ]:
        return self.called_creditations


class DictChoiceFactory:
    """Factory to generate a dictionary choice class"""

    dict_choice: Dict[str, Callable] = {}
    choice_arguments: Dict[str, _ABCDataClass] = {}
    accreditation: Accreditation = Accreditation()

    @classmethod
    def register_choice(
        cls,
        call_name: str,
        author: Optional[str] = None,
        github_handle: Optional[str] = None,
        additional_information: Optional[str] = None,
        credit_type: CreditType = CreditType.NONE,
    ) -> Callable:
        """Registers choices by string with a decorator.
        Optional: registers accreditation.
        """

        def wrapper(callable_cls: Any) -> Any:
            if author or github_handle or additional_information:
                cls.accreditation.register_accreditation(
                    call_name,
                    author,
                    github_handle,
                    additional_information,
                    credit_type,
                )
            cls.dict_choice[call_name] = callable_cls
            return callable_cls

        return wrapper

    @classmethod
    def register_prebuilt_choice(
        cls,
        call_name: str,
        callable_cls: Any,
        author: Optional[str] = None,
        github_handle: Optional[str] = None,
        additional_information: Optional[str] = None,
        credit_type: CreditType = CreditType.NONE,
    ) -> Any:
        """Registers choices by string directly.
        Optional: registers accreditation.
        """
        if author or github_handle or additional_information:
            cls.accreditation.register_accreditation(
                call_name,
                author,
                github_handle,
                additional_information,
                credit_type,
            )
        cls.dict_choice[call_name] = callable_cls

    @classmethod
    def register_arguments(cls, call_name: str) -> Callable:
        def wrapper(argument_class: _ABCDataClass) -> Any:
            cls.choice_arguments[call_name] = argument_class
            return argument_class

        return wrapper

    @classmethod
    def get_arguments(cls, call_name: str) -> _ABCDataClass:
        return cls.choice_arguments[call_name]

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
    def get_accreditations(
        cls,
    ) -> Dict[
        str,
        Tuple[Tuple[Optional[str], Optional[str], Optional[str]], CreditType],
    ]:
        return cls.accreditation.return_accreditations()

    @classmethod
    def view_accreditations(
        cls,
    ) -> None:
        creditations = cls.accreditation.return_accreditations()
        for key, value in creditations.items():
            print(
                f"Object {key}: Author - {value[0][0]}, Github - {value[0][1]}"
                + f", Info - {value[0][2]}\n\t Credit Type {value[1]}"
            )
