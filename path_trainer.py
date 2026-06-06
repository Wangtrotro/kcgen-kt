"""
Path-aware Trainer
==================
Extends trainer.py with path-conditioned KC activation.
The key difference: KC mastery is weighted by path activation before
being injected into the LLM prompt embeddings.

Core idea:
  - Original: all KCs for a problem are activated equally
  - Path-aware: only KCs relevant to the student's solution path are fully activated;
    others are down-weighted (soft) or masked (hard)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from trainer import predict_mastery_level


def path_aware_generator_step(idx, batch, model, lstm, tokenizer, optimizers=None, optimizers_lstm=None,
                              configs=None, train_dl_len=None, train=True, scheduler=None, device=None,
                              group_size=2, multitask=False, predictor=None, pred_loss_fn=None,
                              optimizers_multitask=None, kc_loss_fn=None, trans_linear=None,
                              optimizers_trans=None, path_mode='hard'):
    """
    Path-aware training step. Extends generator_step with path-conditioned KC activation.
    
    Args:
        path_mode: 'hard' (binary mask), 'soft' (weighted), or 'none' (original behavior)
    """
    eps = 1e-8
    if train:
        assert optimizers is not None
        assert optimizers_lstm is not None
        model.train()
        lstm.train()
        if multitask:
            predictor.train()
    else:
        model.eval()
        lstm.eval()
        if multitask:
            predictor.eval()

    # Unpack batch — note the extra fields (path_ids, path_kc_weights) at the end
    (padded_scores, padded_inputs, padded_input_ids_ls, padded_attention_mask_ls,
     padded_labels_ls, padded_prompt_id_lens_ls, padded_kc, level_loc_ls,
     padded_path_ids, path_kc_weights) = batch[0][1:], batch[1][:-1], batch[2][:, 1:], \
        batch[3][1:], batch[4][1:], batch[5][1:], batch[6][1:], batch[7], batch[8][1:], batch[9][1:]

    range_tensor = torch.arange(padded_labels_ls.size(2), device=device).unsqueeze(0).unsqueeze(0)
    range_tensor = range_tensor.repeat(padded_labels_ls.size(0), padded_labels_ls.size(1), 1)
    mask_tensor = (range_tensor >= padded_prompt_id_lens_ls.unsqueeze(-1))

    # Path-conditioned input weight update
    input_wte, kc_prod = path_aware_update_input_weight(
        padded_input_ids_ls, padded_inputs, padded_kc, level_loc_ls, device,
        model, lstm, tokenizer, configs.transition, trans_linear, configs.kc_loss_method,
        path_kc_weights=path_kc_weights, path_mode=path_mode
    )

    T, B, max_length, D = input_wte.shape
    input_wte = input_wte.reshape((T * B), max_length, D)
    padded_attention_mask = padded_attention_mask_ls.reshape((T * B), -1)
    padded_label = padded_labels_ls.reshape((T * B), max_length)

    input_wte_groups = torch.split(input_wte, group_size)
    attention_mask_groups = torch.split(padded_attention_mask, group_size)
    label_groups = torch.split(padded_label, group_size)

    padded_scores = torch.unsqueeze(padded_scores, -1)
    padded_scores = padded_scores.reshape((T * B), -1)
    score_groups = torch.split(padded_scores, group_size)

    kc_prod = torch.unsqueeze(kc_prod, -1)
    kc_prod = kc_prod.reshape((T * B), -1)
    kc_prod_groups = torch.split(kc_prod, group_size)

    padded_mask = mask_tensor.reshape((T * B), -1)
    mask_groups = torch.split(padded_mask, group_size)

    pred_cum_loss, cum_loss, kc_cum_loss = 0.0, 0.0, 0.0
    pred_cnt, cum_cnt, kc_cnt = 0, 0, 0

    pred_total = torch.tensor([]).to(device)
    gt_total = torch.tensor([]).to(device)
    logits_total = torch.tensor([]).to(device)

    for i in range(len(input_wte_groups)):
        input_wte_sub = input_wte_groups[i]
        attention_mask_sub = attention_mask_groups[i]
        label_sub = label_groups[i]

        if train:
            outputs = model(inputs_embeds=input_wte_sub, attention_mask=attention_mask_sub,
                          labels=label_sub, output_hidden_states=True, return_dict=True)
            if multitask:
                mask_sub = mask_groups[i]
                hidden_states = outputs['hidden_states'][-1]
                mask_expand = torch.unsqueeze(mask_sub, -1)
                hidden_states_question = hidden_states * ~mask_expand
                pooled_out = hidden_states_question.sum(dim=1)
                ques_cnt = torch.sum(~mask_expand, dim=1)
                pooled_out = pooled_out / (ques_cnt + eps)
                logits = predictor(pooled_out)
        else:
            with torch.no_grad():
                outputs = model(inputs_embeds=input_wte_sub, attention_mask=attention_mask_sub,
                              labels=label_sub, output_hidden_states=True, return_dict=True)
            if multitask:
                mask_sub = mask_groups[i]
                hidden_states = outputs['hidden_states'][-1]
                mask_expand = torch.unsqueeze(mask_sub, -1)
                hidden_states_question = hidden_states * ~mask_expand
                pooled_out = hidden_states_question.sum(dim=1)
                ques_cnt = torch.sum(~mask_expand, dim=1)
                pooled_out = pooled_out / (ques_cnt + eps)
                logits = predictor(pooled_out)

        loss = outputs["loss"]
        valid_token_cnt = attention_mask_sub.sum()
        cum_loss += loss * valid_token_cnt
        cum_cnt += valid_token_cnt

        score_sub = score_groups[i]
        kc_prod_sub = kc_prod_groups[i]
        kc_loss_sub = kc_loss_fn(kc_prod_sub[score_sub != -100], score_sub[score_sub != -100]).sum()
        kc_cum_loss += kc_loss_sub
        kc_cnt += score_sub[score_sub != -100].shape[-1]

        if multitask:
            if configs.binary_loss_fn == 'BCE':
                gt_total = torch.cat((gt_total, score_sub), 0)
                pred = (torch.sigmoid(logits) > 0.5) * 1
                pred_total = torch.cat((pred_total, pred), 0)
                logits_total = torch.cat((logits_total, logits), 0)
                pred_loss_sub = pred_loss_fn(logits[score_sub != -100], score_sub[score_sub != -100]).sum()
                pred_cum_loss += pred_loss_sub
                pred_cnt += logits[score_sub != -100].shape[-1]
            else:
                score_sub_flat = score_sub.view(-1)
                score_mask = score_sub_flat != -100
                valid_logits = logits[score_mask] / configs.temperature
                valid_scores = score_sub_flat[score_mask].long()
                pred = torch.argmax(logits, dim=-1)
                gt_total = torch.cat((gt_total, score_sub_flat), 0)
                pred_total = torch.cat((pred_total, pred), 0)
                logits_total = torch.cat((logits_total, logits), 0)
                pred_loss_sub = pred_loss_fn(valid_logits, valid_scores).sum()
                pred_cum_loss += pred_loss_sub
                pred_cnt += valid_scores.shape[0]

    kc_cum_loss = kc_cum_loss / kc_cnt
    if multitask:
        pred_cum_loss = pred_cum_loss / pred_cnt
    cum_loss = cum_loss / cum_cnt

    total_loss = cum_loss + kc_cum_loss + pred_cum_loss

    if multitask:
        if configs.kc_loss:
            norm_loss = configs.alpha * (cum_loss / (cum_loss.detach() + eps) + pred_cum_loss / (pred_cum_loss.detach() + eps)) + \
                        (1 - configs.alpha) * kc_cum_loss / (kc_cum_loss.detach() + eps)
        else:
            back_loss = cum_loss + pred_cum_loss
    else:
        back_loss = total_loss

    if train:
        if configs.kc_loss:
            norm_loss.backward()
        else:
            back_loss.backward()

    if train:
        if (idx + 1) % configs.accum_iter == 0 or idx == train_dl_len - 1:
            for optimizer in optimizers:
                optimizer.step()
            if configs.use_scheduler:
                scheduler.step()
            for optimizer in optimizers:
                optimizer.zero_grad()
            for optimizer in optimizers_lstm:
                optimizer.step()
            for optimizer in optimizers_lstm:
                optimizer.zero_grad()
            if configs.transition:
                for optimizer in optimizers_trans:
                    optimizer.step()
                for optimizer in optimizers_trans:
                    optimizer.zero_grad()
            if multitask:
                for optimizer in optimizers_multitask:
                    optimizer.step()
                for optimizer in optimizers_multitask:
                    optimizer.zero_grad()

    log = {
        'loss': total_loss.cpu().detach(),
        'kc_loss': kc_cum_loss.cpu().detach(),
        'generator_loss': cum_loss.cpu().detach()
    }

    if configs.multitask:
        log['predictor_loss'] = pred_cum_loss.cpu().detach()
        pred_res = pred_total[gt_total != -100].detach().cpu() == gt_total[gt_total != -100].detach().cpu()
        log['acc'] = pred_res
        if configs.binary_loss_fn == 'BCE':
            log['auc'] = {'logits': logits_total[gt_total != -100].detach().cpu(),
                         'scores': gt_total[gt_total != -100].detach().cpu()}
        else:
            logits_filtered = logits_total[gt_total != -100]
            scores_filtered = gt_total[gt_total != -100]
            probs = torch.softmax(logits_filtered, dim=-1)[:, 1]
            log['auc'] = {'logits': probs.detach().cpu(), 'scores': scores_filtered.detach().cpu().long()}

    return log


def path_aware_update_input_weight(padded_input_ids_ls, padded_inputs, padded_kc, level_loc_ls,
                                   device, model, lstm, tokenizer, trans=False, trans_linear=None,
                                   kc_loss_method='mean', path_kc_weights=None, path_mode='hard'):
    """
    Extended update_input_weight with path-conditioned KC activation.
    
    path_kc_weights: tensor (T, B, max_kc_len) — per-KC activation weight based on path
    path_mode:
      - 'hard': binary mask — KC either fully active or zeroed
      - 'soft': weighted — KC mastery multiplied by path weight
      - 'none': original behavior (all KCs equally active)
    """
    true_token = tokenizer.convert_tokens_to_ids('True')
    false_token = tokenizer.convert_tokens_to_ids('False')

    if isinstance(model, DDP):
        generator_input_wte = model.module.base_model.model.model.embed_tokens(padded_input_ids_ls)
        true_emb = model.module.base_model.model.model.embed_tokens(torch.tensor(true_token))
        false_emb = model.module.base_model.model.model.embed_tokens(torch.tensor(false_token))
    else:
        generator_input_wte = model.base_model.model.model.embed_tokens(padded_input_ids_ls)
        true_emb = model.base_model.model.model.embed_tokens(torch.tensor(true_token))
        false_emb = model.base_model.model.model.embed_tokens(torch.tensor(false_token))

    result_kc, kc = predict_mastery_level(padded_inputs, padded_kc, lstm, trans, trans_linear)

    # Apply path conditioning to KC mastery
    if path_mode != 'none' and path_kc_weights is not None:
        # path_kc_weights: (T, B, max_kc_len)
        # result_kc: (B, T, max_kc_len)
        path_weights_bt = path_kc_weights.transpose(0, 1)  # → (B, T, max_kc_len)

        # Ensure dimensions match
        if path_weights_bt.shape[2] < result_kc.shape[2]:
            pad_size = result_kc.shape[2] - path_weights_bt.shape[2]
            path_weights_bt = torch.nn.functional.pad(path_weights_bt, (0, pad_size), value=1.0)
        elif path_weights_bt.shape[2] > result_kc.shape[2]:
            path_weights_bt = path_weights_bt[:, :, :result_kc.shape[2]]

        if path_mode == 'hard':
            # Binary: KC is either fully active or contributes nothing
            path_mask = (path_weights_bt > 0.5).float()
            # Don't mask padded positions (where result_kc == -1)
            padding_mask = result_kc == -1
            result_kc_masked = result_kc * path_mask
            result_kc_masked[padding_mask] = -1
            result_kc = result_kc_masked
        elif path_mode == 'soft':
            # Soft weighting: multiply mastery by path relevance
            padding_mask = result_kc == -1
            result_kc_weighted = result_kc * path_weights_bt
            result_kc_weighted[padding_mask] = -1
            result_kc = result_kc_weighted

    B, T = padded_input_ids_ls.shape[0], padded_input_ids_ls.shape[1]
    input_wte = generator_input_wte.clone()
    kc_prod_total = torch.zeros(B, T, requires_grad=True).to(device)
    kc_prod_total = kc_prod_total.clone()

    for i in range(B):
        sent_ind_i, token_ind_i = level_loc_ls[i]
        res_kc_i = result_kc[i]
        input_weight_copy = input_wte[i]
        kc_prod_i = kc_prod_total[i]

        position_dict = {}
        for s, t in zip(sent_ind_i.tolist(), token_ind_i.tolist()):
            if s not in position_dict:
                position_dict[s] = ([], [])
            position_dict[s][0].append(s)
            position_dict[s][1].append(t)

        for key_sent_ind, (sub_sent_ind, sub_token_ind) in position_dict.items():
            sub_sent_ind = torch.tensor(sub_sent_ind)
            sub_token_ind = torch.tensor(sub_token_ind)
            k = sub_token_ind.size(0)
            selected_values = res_kc_i[key_sent_ind, :k]

            # For path-aware: filter out zeroed KCs from the product/mean
            active_mask = selected_values > 0
            active_values = selected_values[active_mask]

            if len(active_values) == 0:
                # No active KCs for this path — use neutral value
                kc_prod_i[key_sent_ind] = 0.5
            elif kc_loss_method == 'prod':
                kc_prod_i[key_sent_ind] = active_values.prod()
            elif kc_loss_method == 'mean':
                kc_prod_i[key_sent_ind] = active_values.mean()
            else:
                kc_prod = active_values.prod()
                kc_prod_i[key_sent_ind] = torch.pow(kc_prod, 1 / len(active_values))

            true_weighted = selected_values.unsqueeze(-1) * true_emb
            false_weighted = (1 - selected_values.unsqueeze(-1)) * false_emb
            new_values = true_weighted + false_weighted
            input_weight_copy[sub_sent_ind, sub_token_ind] = new_values

    input_wte = torch.transpose(input_wte, 0, 1)
    kc_prod_tensor = torch.transpose(kc_prod_total, 0, 1)

    return input_wte, kc_prod_tensor
