from coding_framework_MVP.framework.framework import ModelFramework
from coding_framework_MVP.framework.loops import CustomLoop
from coding_framework_MVP.models.modules.loss import LossChoice
from coding_framework_MVP.models.optimizers.optimizers import OptimizerChoice
from coding_framework_MVP.models.seq2seq import Seq2Seq
from coding_framework_MVP.utils.device import Device
from coding_framework_MVP.utils.initiator import ParameterInitialization


class RetroLoop(CustomLoop):
    pass


def main():
    device = Device(environment.device)
    device.display()

    Arguments

    model = Seq2Seq(
        encoder=encoder,
        tgt_vocab_size=tokenizer.VOCAB_SIZE,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        emb_size=config.EMB_SIZE,
        num_heads=config.NHEAD,
        dim_feedforward=config.FFN_HID_DIM,
        dropout=config.DROPOUT,
        weight_sharing=config.SHARE_WEIGHT,
        pad_idx=config.PAD_IDX,
        max_seq_len=tokenizer.MAX_TENSOR_LEN,
    )
    param_init = ParameterInitialization(method="xavier")
    model = param_init.initialize_model(model)

    loss = LossChoice.get_choice("ce_loss")
    optimizer = OptimizerChoice.get_choice("adam")

    framework = ModelFramework(
        loss=loss,
        metrics=None,
        optimizer=optimizer,
        scheduler=None,
        loggers=None,
    )
    framework.set_model(model)

    Loop = {
        Fit_loop,
        Test_loop,
        Predict_loop,
    }

    trainer = Trainer(...)
    trainer.fit_loop = CustomFitLoop()

    # fit() now uses the new FitLoop!
    trainer.fit(...)

    # the equivalent for validate()
    val_loop = CustomValLoop()
    trainer = Trainer()
    trainer.validate_loop = val_loop
    trainer.validate(...)

    Trainer = {Save, Load}
