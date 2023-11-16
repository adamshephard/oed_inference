import torch.optim as optim

from run_utils.callbacks.base import (
    AccumulateRawOutput,
    PeriodicSaver,
    ProcessAccumulatedRawOutput,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    VisualizeOutput,
    TriggerEngine,
)
from run_utils.callbacks.logging import LoggingEpochOutput, LoggingGradient
from run_utils.engine import Events
from ..targets import gen_targets, prep_sample
from .net_desc import create_model
from ..run_desc import (
    proc_valid_step_output, 
    train_step, 
    valid_step, 
    viz_step_output
)


def get_config(
    mode,
    nr_layers, #nr_classes, 
    nr_types,
    nr_vendors, # added
    seg_mode, 
    # class_info, 
    # class_weights,
    model_name, 
    backbone_name,
    pretrained,
    input_shape,
    output_shape,
    learning_rate,
    train_batch_size,
    valid_batch_size,
    num_workers_train,
    num_workers_valid,
    reduce_epochs, 
    # logdir,
    contour_ksize,
    # merge_classes
    ):
    return {
    "phase_list": [
        {
            "run_info": {
                # may need more dynamic for each network
                "net": {
                    "model_name": model_name,
                    "desc": lambda: create_model(
                        input_ch=3,
                        encoder_backbone_name=backbone_name,
                        pretrained=pretrained, 
                        #nr_classes=nr_classes
                        nr_layers=nr_layers,
                        nr_vendors=nr_vendors, # added
                        freeze=True,
                        ),
                    "optimizer": [
                        optim.Adam,
                        {  # should match keyword for parameters within the optimizer
                            "lr": learning_rate,  # initial learning rate,
                            "betas": (0.9, 0.999),
                        },
                    ],
                    # learning rate scheduler
                    "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, reduce_epochs, gamma=0.05),
                    "extra_info": {
                        "loss": {'ls': {"ce": 1, "dice": 1}},
                    },
                    # path to load, -1 to auto load checkpoint from previous phase,
                    # None to start from scratch
                    "pretrained": None,
                },
            },
            # size_contour only used in instance segmentation
            "target_info": {
                "gen": (gen_targets, {
                    'seg_mode': seg_mode,
                    'model_name': model_name,
                    'nr_types': nr_types,
                    'nr_layers': nr_layers,
                    'contour_ksize': contour_ksize
                    }), 
                "viz": (prep_sample, {'seg_mode': seg_mode, 'nr_classes': nr_layers})}, #nr_classes})},
            "batch_size": {
                "train": train_batch_size,
                "valid": valid_batch_size,
            },
            "backbone_name": backbone_name,
            # "class_info": class_info,
            # "class_weights": class_weights,
            "patch_input_shape": input_shape,
            "patch_output_shape": output_shape,
            "nr_layers": nr_layers,
            "nr_types": nr_types,
            "seg_mode": seg_mode,
            "nr_epochs": 20,
            "contour_ksize": contour_ksize,
        },
        {
            "run_info": {
                # may need more dynamic for each network
                "net": {
                    "model_name": model_name,
                    "desc": lambda: create_model(
                        input_ch=3,
                        encoder_backbone_name=backbone_name,
                        pretrained=pretrained, 
                        #nr_classes=nr_classes
                        nr_layers=nr_layers,
                        nr_vendors=nr_vendors, # added
                        freeze=False,
                        ),
                    "optimizer": [
                        optim.Adam,
                        {  # should match keyword for parameters within the optimizer
                            "lr": learning_rate,  # initial learning rate,
                            "betas": (0.9, 0.999),
                        },
                    ],
                    # learning rate scheduler
                    "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, reduce_epochs, gamma=0.05),
                    "extra_info": {
                        "loss": {'ls': {"ce": 1, "dice": 1}},
                    },
                    # path to load, -1 to auto load checkpoint from previous phase,
                    # None to start from scratch
                    "pretrained": -1,
                },
            },
            # size_contour only used in instance segmentation
            "target_info": {
                "gen": (gen_targets, {
                    'seg_mode': seg_mode,
                    'model_name': model_name,
                    'nr_types': nr_types,
                    'nr_layers': nr_layers,
                    'contour_ksize': contour_ksize
                    }), 
                "viz": (prep_sample, {'seg_mode': seg_mode, 'nr_classes': nr_layers})}, #nr_classes})},
            "batch_size": {
                "train": train_batch_size,
                "valid": valid_batch_size,
            },
            "backbone_name": backbone_name,
            # "class_info": class_info,
            # "class_weights": class_weights,
            "patch_input_shape": input_shape,
            "patch_output_shape": output_shape,
            "nr_layers": nr_layers,
            "nr_types": nr_types,
            "seg_mode": seg_mode,
            "nr_epochs": 30,
            "contour_ksize": contour_ksize,
        }
    ],
    # ------------------------------------------------------------------
    "run_engine": {
            "train": {
                # TODO: align here, file path or what? what about CV?
                "dataset": "",  # whats about compound dataset ?
                "nr_procs": num_workers_train,  # number of threads for dataloader
                "run_step": train_step,  # TODO: function name or function variable ?
                "reset_per_run": False,
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(), # TODO: very slow, may be due to back forth of tensor/numpy ?
                        ScalarMovingAverage(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(),
                        VisualizeOutput(
                           lambda a: viz_step_output(a, model_name, nr_types=nr_types, nr_layers=nr_layers, nr_domains=nr_vendors)
                        ),
                        LoggingEpochOutput(),
                        TriggerEngine("valid"),
                        ScheduleLr(),
                    ],
                },
            },
            "valid": {
                "dataset": "",  # whats about compound dataset ?
                "nr_procs": num_workers_valid,  # number of threads for dataloader
                "run_step": valid_step,
                "reset_per_run": True,  # * to stop aggregating output etc. from last run
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                    Events.EPOCH_COMPLETED: [
                        # TODO: is there way to preload these ?
                        ProcessAccumulatedRawOutput(
                            lambda a: proc_valid_step_output(a, model_name, seg_mode, nr_types=nr_types, nr_layers=nr_layers, nr_domains=nr_vendors)
                        ),
                        LoggingEpochOutput(),
                    ],
                },
            },
        },
    }
