from pytorch_lightning.callbacks import Callback

from .loading_saving import save_model


class CustomSavingCallback(Callback):
    def __init__(
        self, model_save_path: str, save_weights: bool = True
    ) -> None:
        super().__init__()
        self.model_save_path = model_save_path
        self.save_weights = save_weights

    def on_validation_epoch_end(self, trainer, *args, **kwargs) -> None:
        save_model(
            model=trainer.model.model,
            name=f"model-epoch={trainer.current_epoch}",
            path=self.model_save_path,
            save_weights=self.save_weights,
        )
