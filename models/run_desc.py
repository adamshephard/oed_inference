import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss
from sklearn.metrics import f1_score

from collections import OrderedDict


####
def train_step(batch_data, run_info, model_name, seg_mode, nr_types, nr_layers, nr_domains=None):
    # TODO: synchronize the attach protocol
    run_info, state_info = run_info
    seg_mode = seg_mode[0]
    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
        "ce": xentropy_loss,
    }
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    ####
    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]

    ####
    imgs = batch_data["img"]

    if model_name in ["hovernet", "hovernetplus"]:
        if model.module.nr_types is not None or model.module.nr_layers is None:
            true_np = batch_data["np_map"]
            true_hv = batch_data["hv_map"]
    elif seg_mode == 'instance' and model_name == 'unet':
    #     true_np = batch_data["np_map"]
    # else:
    #     if seg_mode == 'instance':
            true_inst = batch_data["inst"]
            true_inst = torch.squeeze(true_inst).to("cuda")
            # true_inst = torch.argmax(true_inst, dim=-1)

    imgs = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs = imgs.permute(0, 3, 1, 2).contiguous() #?

    true_dict = {}

    # HWC
    if model.module.nr_layers is not None:
        true_ls = batch_data["ls_map"]
        true_ls = torch.squeeze(true_ls).to("cuda").type(torch.int64)
        true_ls_onehot = F.one_hot(true_ls, num_classes=model.module.nr_layers)
        true_ls_onehot = true_ls_onehot.type(torch.float32)
        true_dict["ls"] = true_ls_onehot

    if model_name in ["hovernet", "hovernetplus"]:
        if model.module.nr_types is not None or model.module.nr_layers is None:
            true_np = torch.squeeze(true_np).to("cuda").type(torch.int64)
            true_hv = torch.squeeze(true_hv).to("cuda").type(torch.float32)
            true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
            true_dict["np"] = true_np_onehot
            true_dict["hv"] = true_hv

        if model.module.nr_types is not None:
            true_tp = batch_data["tp_map"]
            true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
            true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nr_types)
            true_tp_onehot = true_tp_onehot.type(torch.float32)
            true_dict["tp"] = true_tp_onehot
    
    # elif seg_mode == 'instance' and model_name == 'unet':
        # true_np = torch.squeeze(true_np).to("cuda").type(torch.int64)
        # true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
        # true_dict["np"] = true_np_onehot
    #     true_tp = batch_data["tp_map"]
    #     true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
    #     true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nr_types)
    #     true_tp_onehot = true_tp_onehot.type(torch.float32)
    #     true_dict["tp2"] = true_tp_onehot


    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate

    pred_dict = model(imgs)

    # domain adversarial training # added
    if model.module.nr_vendors is not None:
        pred_vendor = pred_dict.pop("vendor")
        pred_vendor = F.softmax(pred_vendor, dim=-1)
        true_vendor = batch_data["vendor"]

    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )

    if model.module.nr_layers is not None:
        pred_dict["ls"] = F.softmax(pred_dict["ls"], dim=-1)
    
    if model_name in ["hovernet", "hovernetplus"]:
        if model.module.nr_types is not None or model.module.nr_layers is None:
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
        if model.module.nr_types is not None:
            pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)
    elif seg_mode == 'instance' and model_name == 'unet':
        pred_inst = pred_dict['inst']
        pred_inst_swap = pred_inst
    #     # pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    #     pred_dict["tp2"] = F.softmax(pred_dict["tp2"], dim=-1)

    ####
    loss = 0
    loss_opts = run_info["net"]["extra_info"]["loss"]
    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            if model_name == 'unet' and seg_mode == 'instance':
                if loss_name == 'bce':
                    class_weights = torch.Tensor([i for i in range(3)]).to("cuda").type(torch.float32)
                    loss_args = [true_inst, pred_inst_swap]
                elif loss_name == 'dice' or loss_name == 'jaccard':
                    # print('true_inst', true_inst.shape)
                    # true_inst_onehot = F.one_hot(true_inst, num_classes=3)
                    # print('true_inst_hot', true_inst_onehot.shape)
                    # true_inst_onehot = torch.permute(true_inst_onehot, (0, 3, 1, 2))
                    # loss_args = [true_inst_onehot, F.softmax(pred_inst_swap.copy(), dim=1)]                   
                    loss_args = [true_inst, F.softmax(pred_inst_swap, dim=1)]
            else:
                loss_args = [true_dict[branch_name], pred_dict[branch_name]]
                if loss_name == "msge":
                    loss_args.append(true_np_onehot[..., 1])
            term_loss = loss_func(*loss_args)
            track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
            loss += loss_weight * term_loss

    # domain adversarial # added
    if model.module.nr_vendors is not None:
        loss_func = loss_func_dict['bce']
        loss_args = [true_vendor.to("cuda"), pred_vendor.T]
        dom_loss = loss_func(*loss_args)
        track_value("loss_domain_bce", dom_loss.cpu().item())
        loss += dom_loss

    track_value("overall_loss", loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    ####

    # pick 2 random sample from the batch for visualization
    if model_name == 'unet' and seg_mode == 'instance':
        sample_indices = torch.randint(0, true_inst.shape[0], (2,))
    else:
        if model.module.nr_types is not None or model.module.nr_layers is None:
            sample_indices = torch.randint(0, true_np.shape[0], (2,))
        if model.module.nr_types is None and model.module.nr_layers is not None:
            sample_indices = torch.randint(0, true_ls.shape[0], (2,))

    imgs = (imgs[sample_indices]).byte()  # to uint8
    imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    # if model_name == 'unet' and seg_mode == 'instance':
        # if model.module.nr_types is not None or model.module.nr_layers is None:
        #     pred_dict["np"] = pred_dict["np"][..., 1]  # return pos only
        # pred_dict['inst'] = (pred_dict['inst'].copy()[sample_indices]).detach().cpu().numpy()

    pred_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in pred_dict.items()
    }

    true_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in true_dict.items()
    }

    if model_name == 'unet' and seg_mode == 'instance':
    #     if model.module.nr_types is not None or model.module.nr_layers is None:
    #         true_dict["np"] = true_np
        true_dict['inst'] = true_inst

    result_dict["raw"] = {  # protocol for contents exchange within `raw`
        "img": imgs,
    }

    # * Its up to user to define the protocol to process the raw output per step!
    if model_name == 'unet' and seg_mode == 'instance':
        result_dict["raw"]["inst"] = (true_dict["inst"], pred_dict["inst"])
    else:
        if model.module.nr_types is not None or model.module.nr_layers is None:
            result_dict["raw"]["np"] = (true_dict["np"], pred_dict["np"])
            if 'hovernet' in model_name:
                result_dict["raw"]["hv"] = (true_dict["hv"], pred_dict["hv"])

        if model.module.nr_layers is not None:
            result_dict["raw"]["ls"] = (true_dict["ls"], pred_dict["ls"])

        if model.module.nr_types is not None:
            result_dict["raw"]["tp"] = (true_dict["tp"], pred_dict["tp"])
    
    # domain adversarial # added
    if model.module.nr_vendors is not None:
        result_dict["raw"]["vendor"] = (true_vendor.detach().cpu().numpy(), pred_vendor.detach().cpu().numpy())

    return result_dict


####
def valid_step(batch_data, run_info, model_name, seg_mode, nr_types, nr_layers):
    run_info, state_info = run_info
    seg_mode = seg_mode[0]
    ####
    model = run_info["net"]["desc"]
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    if model_name == 'unet' and seg_mode == 'instance':
        true_inst = batch_data["inst"]
    else:
        if model.module.nr_types is not None or model.module.nr_layers is None:
            true_np = batch_data["np_map"]
            if 'hovernet' in model_name:
                true_hv = batch_data["hv_map"]

    imgs_gpu = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous() #?

    true_dict = {}

    # HWC
    if model_name == 'unet' and seg_mode == 'instance':
        true_dict['inst'] = true_inst.to("cuda").type(torch.int64)
    else:
        if model.module.nr_types is not None or model.module.nr_layers is None:
            true_np = true_np.to("cuda").type(torch.int64)
            # true_np = torch.squeeze(true_np).to("cuda").type(torch.int64)
            true_dict["np"] = true_np
            if 'hovernet' in model_name:
                true_hv = true_hv.to("cuda").type(torch.float32)
                # true_hv = torch.squeeze(true_hv).to("cuda").type(torch.float32)
                true_dict["hv"] = true_hv
        
        if model.module.nr_types is not None:
            true_tp = batch_data["tp_map"]
            # true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
            true_tp = true_tp.to("cuda").type(torch.int64)
            true_dict["tp"] = true_tp

        if model.module.nr_layers is not None:
            true_ls = batch_data["ls_map"]
            # true_ls = torch.squeeze(true_ls).to("cuda").type(torch.int64)
            true_ls = true_ls.to("cuda").type(torch.int64)
            true_dict["ls"] = true_ls

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(imgs_gpu)

        # domain adversarial training # added
        if model.module.nr_vendors is not None:
            pred_vendor = pred_dict.pop("vendor")
            pred_vendor = F.softmax(pred_vendor, dim=-1)
            true_vendor = batch_data["vendor"]

        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()] #?
        )

        if model_name == 'unet' and seg_mode == 'instance':
            pred_dict['inst'] = F.softmax(pred_dict["inst"], dim=-1)
        else:
            if model.module.nr_types is not None or model.module.nr_layers is None:
                pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]
            if model.module.nr_types is not None:
                type_map = F.softmax(pred_dict["tp"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=False)
                type_map = type_map.type(torch.float32)
                pred_dict["tp"] = type_map
            if model.module.nr_layers is not None:
                type_map = F.softmax(pred_dict["ls"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=False)
                type_map = type_map.type(torch.float32)
                pred_dict["ls"] = type_map


    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {  # protocol for contents exchange within `raw`
                "raw": {
                    "imgs": imgs.numpy(),
                }
            }
    if model_name == 'unet' and seg_mode == 'instance':
        result_dict["raw"]["true_inst"] = true_dict["inst"].cpu().numpy()
        result_dict["raw"]["prob_inst"] = pred_dict["inst"].cpu().numpy()   
    else:     
        if model.module.nr_types is not None or model.module.nr_layers is None:
            result_dict["raw"]["true_np"] = true_dict["np"].cpu().numpy()
            result_dict["raw"]["prob_np"] = pred_dict["np"].cpu().numpy()
            if 'hovernet' in model_name:
                result_dict["raw"]["true_hv"] = true_dict["hv"].cpu().numpy()
                result_dict["raw"]["pred_hv"] = pred_dict["hv"].cpu().numpy() 

        if model.module.nr_types is not None:
            result_dict["raw"]["true_tp"] = true_dict["tp"].cpu().numpy()
            result_dict["raw"]["pred_tp"] = pred_dict["tp"].cpu().numpy()

        if model.module.nr_layers is not None:
            result_dict["raw"]["true_ls"] = true_dict["ls"].cpu().numpy()
            result_dict["raw"]["pred_ls"] = pred_dict["ls"].cpu().numpy()

    # domain adversarial # added
    if model.module.nr_vendors is not None:
        result_dict["raw"]["true_vendor"] = true_vendor.cpu().numpy()
        result_dict["raw"]["pred_vendor"] = pred_vendor.cpu().numpy()
    
    return result_dict


####
def infer_step(batch_data, model): #, model_name, seg_mode, nr_types, nr_layers):
    ####
    patch_imgs = batch_data

    patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32)  # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    ####
    model.eval()  # infer mode

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(patch_imgs_gpu)

        # domain adversarial training # added
        if "vendor" in pred_dict:
            pred_vendor = pred_dict.pop("vendor")
            pred_vendor = F.softmax(pred_vendor, dim=-1)
        else:
            pred_vendor = None

        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        if "np" in pred_dict:
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        if "ls" in pred_dict:
            layer_map = F.softmax(pred_dict["ls"], dim=-1)
            # layer_map = pred_dict["ls"][...,1]
            # layer_map = torch.argmax(layer_map, dim=-1, keepdim=True)
            # layer_map = layer_map.type(torch.float32)
            pred_dict["ls"] = layer_map
        # if pred_vendor is not None:
        #     pred_dict["vendor"] = pred_vendor
        pred_output = torch.cat(list(pred_dict.values()), -1)

    # * Its up to user to define the protocol to process the raw output per step!
    return pred_output.cpu().numpy()


####
def viz_step_output(raw_data, model_name, nr_types=None, nr_layers=None, nr_domains=None, skip=True):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data["img"]

    if skip is True:
        _ = raw_data.pop("img")
        key = list(raw_data.keys())[0]
        pred, true = raw_data[key]
        aligned_shape = [list(imgs.shape), list(true.shape), list(pred.shape)]
        aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]
        # aligned_shape = [list([164, 164])]
        # aligned_shape = np.min(np.array(aligned_shape), axis=0)[0:2]
    else:
        if nr_types is not None and nr_layers is not None:
            true_np, pred_np = raw_data["np"]
            true_tp, pred_tp = raw_data["tp"]
            true_ls, pred_ls = raw_data["ls"]
            aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
            aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]
            if 'hovernet' in model_name:
                true_hv, pred_hv = raw_data["hv"]
        elif nr_types is not None and nr_layers is None:
            true_np, pred_np = raw_data["np"]
            true_tp, pred_tp = raw_data["tp"]
            aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
            aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]
            if 'hovernet' in model_name:
                true_hv, pred_hv = raw_data["hv"]
        elif nr_types is None and nr_layers is not None:
            true_ls, pred_ls = raw_data["ls"]
            # aligned_shape = [list([164, 164])]
            aligned_shape = [list(imgs.shape), list(true_ls.shape), list(pred_ls.shape)]
            # aligned_shape = np.min(np.array(aligned_shape), axis=0)[0:2]
            aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]
        #  if nr_layers is not None and nr_types is None: # removed for
        elif nr_types is None and nr_layers is None: # e.g. for nuclear pixel prediction only
            true_np, pred_np = raw_data["np"]
            aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
            aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]
            if 'hovernet' in model_name:
                true_hv, pred_hv = raw_data["hv"]

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32"))
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)

        true_viz_list = [img]
        # cmap may randomly fails if of other types
        if skip is False:
            if nr_types is not None or nr_layers is None:
                true_viz_list.append(colorize(true_np[idx], 0, 1))
                if 'hovernet' in model_name:
                    true_viz_list.append(colorize(true_hv[idx][..., 0], -1, 1))
                    true_viz_list.append(colorize(true_hv[idx][..., 1], -1, 1))
            if nr_types is not None:  # TODO: a way to pass through external info
                true_viz_list.append(colorize(true_tp[idx], 0, nr_types))
            if nr_layers is not None:
                true_viz_list.append(colorize(true_ls[idx], 0, nr_layers))

        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img]
        # cmap may randomly fails if of other types
        if skip is False:
            if nr_types is not None or nr_layers is None:
                pred_viz_list.append(colorize(pred_np[idx], 0, 1))
                if 'hovernet' in model_name:
                    pred_viz_list.append(colorize(pred_hv[idx][..., 0], -1, 1))
                    pred_viz_list.append(colorize(pred_hv[idx][..., 1], -1, 1))
            if nr_types is not None:
                pred_viz_list.append(colorize(pred_tp[idx], 0, nr_types))
            if nr_layers is not None:
                pred_viz_list.append(colorize(pred_ls[idx], 0, nr_layers))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(np.concatenate([true_viz_list, pred_viz_list], axis=0))
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list


####
from itertools import chain


def proc_valid_step_output(raw_data, model_name, seg_mode, nr_types=None, nr_layers=None, nr_domains=None):
    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    if 'hovernet' in model_name:
        over_inter = 0
        over_total = 0
        over_correct = 0
        if nr_types is not None or nr_layers is None:
            prob_np = raw_data["prob_np"]
            true_np = raw_data["true_np"]
            for idx in range(len(raw_data["true_np"])):
                patch_prob_np = prob_np[idx]
                patch_true_np = true_np[idx]
                patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
                inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
                correct = (patch_pred_np == patch_true_np).sum()
                over_inter += inter
                over_total += total
                over_correct += correct
            nr_pixels = len(true_np) * np.size(true_np[0])
            acc_np = over_correct / nr_pixels
            dice_np = 2 * over_inter / (over_total + 1.0e-8)
            track_value("np_acc", acc_np, "scalar")
            track_value("np_dice", dice_np, "scalar")

        # * TP statistic
        if nr_types is not None:
            pred_tp = raw_data["pred_tp"]
            true_tp = raw_data["true_tp"]
            over_dice_tp = 0
            for type_id in range(0, nr_types):
                over_inter = 0
                over_total = 0
                for idx in range(len(raw_data["true_np"])):
                    patch_pred_tp = pred_tp[idx]
                    patch_true_tp = true_tp[idx]
                    inter, total = _dice_info(patch_true_tp, patch_pred_tp, type_id)
                    over_inter += inter
                    over_total += total
                dice_tp = 2 * over_inter / (over_total + 1.0e-8)
                over_dice_tp += dice_tp
                track_value("tp_dice_%d" % type_id, dice_tp, "scalar")
            mean_dice_tp = over_dice_tp / nr_types
            track_value("tp_mean_dice", mean_dice_tp, "scalar")
        
        # * HV regression statistic
        if nr_types is not None or nr_layers is None:
            pred_hv = raw_data["pred_hv"]
            true_hv = raw_data["true_hv"]

            over_squared_error = 0
            for idx in range(len(raw_data["true_np"])):
                patch_pred_hv = pred_hv[idx]
                patch_true_hv = true_hv[idx]
                squared_error = patch_pred_hv - patch_true_hv
                squared_error = squared_error * squared_error
                over_squared_error += squared_error.sum()
            mse = over_squared_error / nr_pixels
            track_value("hv_mse", mse, "scalar")

    # * LS statistic
    if nr_layers is not None:
        pred_ls = raw_data["pred_ls"]
        true_ls = raw_data["true_ls"]
        over_dice_ls = 0
        for type_id in range(0, nr_layers):
            over_inter = 0
            over_total = 0
            for idx in range(len(raw_data["true_ls"])):
                patch_pred_ls = pred_ls[idx]
                patch_true_ls = true_ls[idx]
                inter, total = _dice_info(patch_true_ls, patch_pred_ls, type_id)
                over_inter += inter
                over_total += total
            dice_ls = 2 * over_inter / (over_total + 1.0e-8)
            over_dice_ls += dice_ls
            track_value("ls_dice_%d" % type_id, dice_ls, "scalar")
        mean_dice_ls = over_dice_ls / nr_layers
        track_value("ls_mean_dice", mean_dice_ls, "scalar")
    
    if model_name == 'unet' and seg_mode == 'instance':
        pred = raw_data["prob_inst"]
        true = raw_data["true_inst"]
        dice_list = []
        # only consider positive classes!
        for class_id in range(1, nr_types):
            over_inter = 0
            over_total = 0
            for idx in range(len(raw_data["prob_inst"])):
                patch_pred = pred[idx]
                # patch_pred = np.argmax(patch_pred, axis=-1)
                patch_true = true[idx]
                inter, total = _dice_info(patch_true, patch_pred, class_id)
                over_inter += inter
                over_total += total
            dice = 2 * over_inter / (over_total + 1.0e-8)
            track_value("Dice_%s" % class_id, dice, "scalar")
            dice_list.append(dice)
        track_value("Dice_Mean", np.mean(dice_list), "scalar")

    ## domain adversarial training ## # added
    if nr_domains is not None:
        pred_vendor = raw_data["pred_vendor"]
        true_vendor = raw_data["true_vendor"]
        pred_vendor = [np.argmax(p) for p in pred_vendor]
        for type_id in range(0, nr_domains):
            over_inter = 0
            over_total = 0
            pred_vendor_ = [1 if p == type_id else 0 for p in pred_vendor]
            true_vendor_ = [1 if t == type_id else 0 for t in true_vendor]
            vendor_f1 = f1_score(true_vendor_, pred_vendor_)
            track_value("domain_f1_%d" % type_id, vendor_f1, "scalar")
        macro_f1 = f1_score(true_vendor, pred_vendor, average="macro")
        track_value("domain_macro_f1", macro_f1, "scalar")

    # *
    imgs = raw_data["imgs"]
    selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
    imgs = np.array([imgs[idx] for idx in selected_idx])
    viz_raw_data = {"img": imgs}
    if nr_types is not None or nr_layers is None:
        true_np = np.array([true_np[idx] for idx in selected_idx])
        prob_np = np.array([prob_np[idx] for idx in selected_idx])
        viz_raw_data["np"] = (true_np, prob_np)
        if 'hovernet' in model_name:
            true_hv = np.array([true_hv[idx] for idx in selected_idx])
            pred_hv = np.array([pred_hv[idx] for idx in selected_idx])
            #viz_raw_data = {"img": imgs, "np": (true_np, prob_np), "hv": (true_hv, pred_hv)}
            viz_raw_data["hv"] = (true_hv, pred_hv)
    if nr_types is not None:
        true_tp = np.array([true_tp[idx] for idx in selected_idx])
        pred_tp = np.array([pred_tp[idx] for idx in selected_idx])
        viz_raw_data["tp"] = (true_tp, pred_tp)
    if nr_layers is not None:
        true_ls = np.array([true_ls[idx] for idx in selected_idx])
        pred_ls = np.array([pred_ls[idx] for idx in selected_idx])
        viz_raw_data["ls"] = (true_ls, pred_ls)
 
    viz_fig = viz_step_output(viz_raw_data, model_name, nr_types, nr_layers, skip=False)
    track_dict["image"]["output"] = viz_fig

    return track_dict
