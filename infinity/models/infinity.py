# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Definition of Infinity transformer model.
"""

import math
import random
import time
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any
import json
import os
import uuid
import cv2
import imageio
import os.path as osp
import functools
import asyncio

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from torch.utils.checkpoint import checkpoint
import numpy as np
from torch.nn.attention.flex_attention import flex_attention

import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import flash_fused_op_installed, SelfAttnBlock, FastRMSNorm
from infinity.models.rope import precompute_rope4d_freqs_grid
from infinity.models.flex_attn_mask import build_flex_attn_func
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index, get_activated_h_div_w_templates
from infinity.models.apg import normalized_guidance
from infinity.utils.sequence_parallel import sp_split_sequence_by_dim, sp_gather_sequence_by_dim, SequenceParallelManager as sp_manager

try:
    from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None


from vlm.video_evaluation_client import quick_evaluate_media

class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C

class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None, scale_ind=None, context_info=None, last_repetition_step=True, ref_text_scale_inds=[]):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule, scale_ind, context_info, last_repetition_step, ref_text_scale_inds, use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule, scale_ind, context_info, last_repetition_step, ref_text_scale_inds)
        return h

def get_timestep_embedding(dim, timesteps=1000, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    assert dim % 2 == 0, "dimension must be even number"
    half = dim // 2
    timesteps = torch.arange(timesteps, dtype=torch.float32)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

def save_video(ndarray_image_list, fps=24, save_filepath='tmp.mp4'):
    if len(ndarray_image_list) == 1:
        save_filepath = save_filepath.replace('.mp4', '.jpg')
        cv2.imwrite(save_filepath, ndarray_image_list[0])
        print(f"Image saved as {osp.abspath(save_filepath)}")
    else:
        h, w = ndarray_image_list[0].shape[:2]
        os.makedirs(osp.dirname(save_filepath), exist_ok=True)
        imageio.mimsave(save_filepath, ndarray_image_list[:, :, :, ::-1], fps=fps,)
        print(f"Video saved as {osp.abspath(save_filepath)}")

def timeit(func):
    """
    计算函数执行时间的装饰器（支持同步/异步）
    优先级: self.logger > print
    """

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start_time

        msg = f"函数 {func.__name__} 执行耗时: {duration:.6f} 秒"

        if args and hasattr(args[0], "logger"):  # 类里有 self.logger
            args[0].logger.info(msg)
        else:
            print(msg)

        return result

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        duration = time.perf_counter() - start_time

        msg = f"函数 {func.__name__} 执行耗时: {duration:.6f} 秒"

        if args and hasattr(args[0], "logger"):
            args[0].logger.info(msg)
        else:
            print(msg)

        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def encode_prompt(t5_path, text_tokenizer, text_encoder, prompt, enable_positive_prompt=False, low_vram_mode=False):
    if enable_positive_prompt:
        pass
    # print(f'prompt={prompt}')
    captions = [prompt]
    if 'flan-t5' in t5_path:
        tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokens.input_ids.cuda(non_blocking=True)
        mask = tokens.attention_mask.cuda(non_blocking=True)
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
        lens: List[int] = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        Ltext = max(lens)
        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    else:
        text_features = text_encoder(captions, 'cuda')
        lens = [len(item) for item in text_features]
        cu_seqlens_k = [0]
        for len_i in lens:
            cu_seqlens_k.append(cu_seqlens_k[-1] + len_i)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
        Ltext = max(lens)
        kv_compact = torch.cat(text_features, dim=0).float()
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple

class Infinity(nn.Module):
    def __init__(
        self, vae_local,
        arch='qwen',                         # var or qwen
        qwen_qkvo_bias=False,               # qwen qwen_qkvo_bias
        text_channels=0, text_maxlen=0,     # text-cond generation
        embed_dim=1024, depth=16,
        num_key_value_heads=-1,
        num_heads=16, mlp_ratio=4.,   # model's architecture
        norm_eps=1e-6, rms_norm=False,      # norm layer
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        drop_path_rate=0.1,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        top_p=0.0,
        top_k=0.0,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        add_lvl_embeding_on_first_block=1,
        num_of_label_value=2,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        apply_spatial_patchify = 0,
        inference_mode=False,
        other_args=None,
    ):
        super().__init__()
        # set hyperparameters
        self.C = embed_dim
        self.vae_embed_dim = vae_local.codebook_dim
        self.detail_scale_min_tokens = other_args.detail_scale_min_tokens
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.codebook_dim * 4
        else:
            self.d_vae = vae_local.codebook_dim
        self.other_args = other_args
        self.mask_type = other_args.mask_type
        self.context_frames = other_args.context_frames
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.video_frames)
        self.num_of_label_value = num_of_label_value
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * self.num_of_label_value) if self.num_of_label_value else vae_local.vocab_size
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.image_batch_size = other_args.image_batch_size
        self.video_batch_size = other_args.video_batch_size
        self.arch = arch
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = get_activated_h_div_w_templates(train_h_div_w_list, self.h_div_w_templates)
        self.video_frames = video_frames


        assert add_lvl_embeding_on_first_block in [0,1]
        self.add_lvl_embeding_on_first_block = add_lvl_embeding_on_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        self.image_scale_repetition = json.loads(other_args.image_scale_repetition)
        self.video_scale_repetition = json.loads(other_args.video_scale_repetition)
        print(f'arch: {arch}, self.pn: {self.pn}, self.codebook_dim: {self.codebook_dim}, self.add_lvl_embeding_on_first_block: {self.add_lvl_embeding_on_first_block}, \
            self.num_of_label_value: {self.num_of_label_value}, self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw} \
            self.train_h_div_w_list: {self.train_h_div_w_list}, self.image_scale_repetition: {self.image_scale_repetition}, self.video_scale_repetition: {self.video_scale_repetition}')
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'

        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)

        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0

        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'

        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0

        # [inp & position embedding]
        self.norm0_cond = nn.Identity()
        self.selecting_idx = None
        self.num_classes = 0
        self.D = self.C

        cfg_uncond = torch.empty(512, self.Ct5)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(0)
        torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
        cfg_uncond /= self.Ct5 ** 0.5
        if rand_uncond:
            self.register_buffer('cfg_uncond', cfg_uncond)
        else:
            self.cfg_uncond = nn.Parameter(cfg_uncond)

        if other_args.simple_text_proj:
            self.text_norm = nn.Identity()
            self.text_proj = nn.Linear(self.Ct5, self.D)
        else:
            self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
            self.text_proj = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )
        self.sos_token = nn.Parameter(torch.empty(1, 1, self.D))

        if self.rope2d_each_sa_layer:
            if other_args.rope_type == '4d':
                tmp_h_div_w_template = self.train_h_div_w_list[0]
                scales_in_one_clip = self.dynamic_resolution_h_w[tmp_h_div_w_template][self.pn]['scales_in_one_clip']
                max_video_scales = self.dynamic_resolution_h_w[tmp_h_div_w_template][self.pn]['max_video_scales']
                if other_args.dynamic_scale_schedule == 'infinity_star_interact':
                    max_scales = 1000
                else:
                    max_scales = sum(self.image_scale_repetition) + sum(self.video_scale_repetition) * (max_video_scales//scales_in_one_clip-1)
                    max_scales = max(max_scales, max_video_scales)
                rope2d_freqs_grid = precompute_rope4d_freqs_grid(dim=self.C//self.num_heads,
                                                                 pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                                                                 activated_h_div_w_templates=self.train_h_div_w_list,
                                                                 steps_per_frame=other_args.steps_per_frame,
                                                                 max_scales=max_scales+10,
                                                                 max_frames=int(self.video_frames/other_args.temporal_compress_rate+1),
                                                                 max_height=1800 // 8, max_width=1800 // 8,
                                                                 text_maxlen=self.text_maxlen,
                                                                 pn=self.pn,
                                                                 args=other_args,)
            else:
                raise ValueError(f'self.rope_type == {self.rope_type} unsupported!')
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')

        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        if self.arch == 'qwen':
            self.norm_hidden_sates = FastRMSNorm(self.C)
        else:
            raise ValueError(f'arch={self.arch} not implemented')

        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        if self.use_flex_attn:
            self.flex_attention = torch.compile(flex_attention)

        self.unregistered_blocks = []
        for _ in range(depth):
            block = SelfAttnBlock(
                embed_dim=self.C,
                cond_dim=self.D,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                mlp_ratio=mlp_ratio,
                use_flex_attn=use_flex_attn,
                pad_to_multiplier=pad_to_multiplier,
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                mask_type=other_args.mask_type,
                context_frames=other_args.context_frames,
                steps_per_frame=other_args.steps_per_frame,
                arch=self.arch,
                qwen_qkvo_bias=qwen_qkvo_bias,
                inject_sync=other_args.inject_sync,
            )
            # block.bfloat16()
            self.unregistered_blocks.append(block)

        # [head]
        self.head = nn.Linear(self.C, self.other_args.detail_scale_dim*self.other_args.num_of_label_value)
        if self.other_args.use_two_stage_lfq:
            self.semantic_head2 = nn.Linear(self.C, self.other_args.semantic_scale_dim*self.other_args.num_of_label_value)

        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, {block_chunks=}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
        print(
            f'    [Infinity config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n',
            end='\n\n', flush=True
        )

    def get_loss_acc(self, x_BLC, sequece_packing_scales, gt):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        if self.arch == 'qwen':
            x_BLC = self.norm_hidden_sates(x_BLC)

        with torch.amp.autocast('cuda', enabled=False):
            x_BLC = x_BLC.float()
            logits_full = self.head(x_BLC)
            if self.other_args.use_two_stage_lfq:
                logits_semantic_full = self.semantic_head2(x_BLC)
                global_token_ptr, global_scale_ptr = 0, 0
                loss_list, acc_list = [], []
                for i in range(len(sequece_packing_scales)):
                    for j in range(len(sequece_packing_scales[i])):
                        pt, ph, pw = sequece_packing_scales[i][j]
                        mul_pt_ph_pw = pt * ph * pw
                        if ph * pw >= self.detail_scale_min_tokens:
                            logits = logits_full[:,global_token_ptr:global_token_ptr+mul_pt_ph_pw]
                        else:
                            logits = logits_semantic_full[:,global_token_ptr:global_token_ptr+mul_pt_ph_pw]
                        logits = logits.reshape(x_BLC.shape[0], mul_pt_ph_pw, -1, self.other_args.num_of_label_value)
                        logits = logits.permute(0,3,1,2) # [1, mul_pt_ph_pw, d, num_of_label_value] -> [1, num_of_label_value, mul_pt_ph_pw, d]
                        # gt[global_scale_ptr]: [1, mul_pt_ph_pw, d]
                        loss_this_scale = F.cross_entropy(logits, gt[global_scale_ptr], reduction='none').mean(-1)[0] # [mul_pt_ph_pw]
                        acc_this_scale = (logits.argmax(1) == gt[global_scale_ptr]).float().mean(-1)[0] # [mul_pt_ph_pw]

                        loss_list.append(loss_this_scale)
                        acc_list.append(acc_this_scale)
                        global_scale_ptr += 1
                        global_token_ptr += mul_pt_ph_pw
                loss_list = torch.cat(loss_list)
                acc_list = torch.cat(acc_list)
            else:
                gt = torch.cat(gt, 1) # [B, L, d]
                logits = logits_full
                logits = logits.reshape(x_BLC.shape[0], x_BLC.shape[1], -1, self.other_args.num_of_label_value)
                logits = logits.permute(0,3,1,2) # [B, num_of_label_value, L, d]
                if self.other_args.num_of_label_value > 1:
                    loss_list = F.cross_entropy(logits, gt, reduction='none').mean(-1)[0] # [L]
                    acc_list = (logits.argmax(1) == gt).float().mean(-1)[0] # [L]
                elif self.other_args.num_of_label_value == 1:
                    loss_list = torch.nn.functional.mse_loss(logits.squeeze(1), gt[global_scale_ptr], reduction='none').mean(-1)[0] # [L]
                    acc_list = loss_list
            return loss_list, acc_list

    def get_logits_during_infer(self, x_BLC, is_semantic_scale):
        if self.arch == 'qwen':
            x_BLC = self.norm_hidden_sates(x_BLC)
        with torch.amp.autocast('cuda', enabled=False):
            x_BLC = x_BLC.float()
            if self.other_args.use_two_stage_lfq:
                if is_semantic_scale:
                    logits = self.semantic_head2(x_BLC)
                else:
                    logits = self.head(x_BLC)
            else:
                logits = self.head(x_BLC)
        return logits

    def pick_visual_tokens(
        self,
        x_BLC,
        sequece_packing_scales,
        visual_tokens_len,
        args,
    ):
        visual_tokens = x_BLC[:,:visual_tokens_len]
        return visual_tokens

    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC: torch.Tensor,
        visual_rope_cache = None,
        sequece_packing_scales = None, # [[(1,1,1)->(5,5,5)], [(1,1,1)->(10,10,10)]] 1LC
        super_scale_lengths = None,
        super_querysid_super_refsid = None,
        other_info_by_scale = None,
        gt_BL = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        :return: logits BLV, V is vocab_size
        """

        x_BLC= x_BLC.float()       # input should be float32
        B = x_BLC.shape[0]
        cond_BD_or_gss, ca_kv = None, None

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            # 12 kv_compact, lens

            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact[0, 0] += must_on_graph
            # drop cond
            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total:total+le] = self.cfg_uncond[:le]
                total += le

            visual_tokens_len = x_BLC.shape[1]
            # forms prefix_tokens
            kv_compact = self.text_norm(kv_compact)
            kv_compact = self.text_proj(kv_compact).contiguous()
            x_BLC = self.word_embed(self.norm0_ve(x_BLC)) # norm0_ve is Identity
            x_BLC = torch.cat((x_BLC, kv_compact.unsqueeze(0)), dim=1)

            if self.other_args.train_with_var_seq_len:
                pad_seq_len = int(np.ceil(x_BLC.shape[1]/self.pad_to_multiplier))*self.pad_to_multiplier - x_BLC.shape[1]
            else:
                pad_seq_len = self.other_args.train_max_token_len - x_BLC.shape[1]
            if pad_seq_len > 0:
                x_BLC = F.pad(x_BLC, (0, 0, 0, pad_seq_len), value=0.0)

            # valid_sequence_ratio = 1 - pad_seq_len / self.other_args.train_max_token_len
            valid_sequence_ratio = 1 - pad_seq_len / x_BLC.shape[1]
            assert self.use_flex_attn
            attn_bias_or_two_vector = None

        attn_fn = build_flex_attn_func(
            flex_attention=self.flex_attention,
            seq_l=x_BLC.shape[1],
            prefix_lens=lens,
            args=self.other_args,
            device=x_BLC.device,
            batch_size=B,
            heads=None,
            pad_seq_len=pad_seq_len,
            sequece_packing_scales=sequece_packing_scales,
            super_scale_lengths=super_scale_lengths,
            super_querysid_super_refsid=super_querysid_super_refsid,
        )

        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(x_BLC.device)
        rope_cache_list = [visual_rope_cache]
        for i in range(len(lens)):
            rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[i]])
        rope_cache = torch.cat(rope_cache_list, dim=4)
        if pad_seq_len > 0:
            rope_cache = F.pad(rope_cache, (0,0,0,pad_seq_len), 'constant', 0.)
        assert rope_cache.shape[4] == x_BLC.shape[1], f'{rope_cache.shape[4]} != {x_BLC.shape[1]}'
        # [2. block loop]
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training

        if sp_manager.sp_on():
            # [B, raw_L, C] --> [B, raw_L/sp_size, C]
            x_BLC = sp_split_sequence_by_dim(x_BLC, 1)

        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, attn_fn, rope_cache, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, rope2d_freqs_grid=rope_cache)
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=rope_cache)

        if sp_manager.sp_on():
            # [B, raw_L/sp_size, C] --> [B, raw_L, C]
            x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)

        # [3. unpad the seqlen dim, and then get logits]
        x_BLC = self.pick_visual_tokens(x_BLC, sequece_packing_scales, visual_tokens_len, self.other_args)
        loss_list, acc_list = self.get_loss_acc(x_BLC, sequece_packing_scales, gt_BL)
        return loss_list, acc_list, valid_sequence_ratio

    def prepare_text_conditions(
        self,
        label_B_or_BLT,
        cfg_list,
        B,
        negative_label_B_or_BLT,
        vae_scale_schedule=None,
        text_token_only=False,
        text_maxlen_this_iter=512,
    ):
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        bs = B
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
                lens = lens + lens
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
                lens = lens + lens_un
        kv_compact = self.text_norm(kv_compact)
        kv_compact = self.text_proj(kv_compact).contiguous()
        assert B == 1
        prefix_tokens = torch.zeros((bs, text_maxlen_this_iter, self.C), dtype=kv_compact.dtype, device=kv_compact.device)
        total = 0
        for i, le in enumerate(lens):
            assert le <= text_maxlen_this_iter
            prefix_tokens[i,:le] = kv_compact[total:total+le]
            total += le
        return prefix_tokens, lens

    @torch.no_grad()
    def autoregressive_infer(
        self,
        args=None,
        **kwargs,
    ):
        if 'infinity_elegant' in args.dynamic_scale_schedule:
            infer_func = self.ar_infer_infinity_elegant
        elif 'infinity_star_interact' in args.dynamic_scale_schedule:
            infer_func = self.ar_infer_infinity_star_interact
        else:
            infer_func = self.autoregressive_infer_cfg
        return infer_func(args=args, **kwargs)

    def embeds_codes2input(
        self,
        last_stage, # [B, d, t, h, w]
        repeat=1,
    ):
        if self.apply_spatial_patchify: # patchify operation
            last_stage = last_stage.permute(0,2,1,3,4) # [B, t, d, 2h, 2w]
            last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, t, 4d, h, w]
            last_stage = last_stage.permute(0,2,1,3,4) # [B, 4d, t, h, w]
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, t*h*w] or [B, 4d, t*h*w]
        last_stage = torch.permute(last_stage, [0,2,1]) # [B, t*h*w, d] or [B, t*h*w, 4d]
        last_stage = self.word_embed(self.norm0_ve(last_stage))
        last_stage = last_stage.repeat(repeat, 1, 1)
        return last_stage

    @torch.no_grad()
    @timeit
    def fast_forward_preview(
        self,
        current_si,
        last_stage,
        summed_codes,
        scale_schedule,
        vae_scale_schedule,
        remaining_scales,
        vae,
        cfg_list,
        tau_list,
        B,
        bs,
        cond_BD_or_gss,
        ca_kv,
        attn_mask,
        block_chunks,
        device,
        args,
        context_info,
        first_full_spatial_size_scale_index,
        get_visual_rope_embeds,
        rng=None,
        ref_text_scale_inds=['t0'],
        use_greedy=True,
        preview_tau=0.1,
        preview_threshold=0.7,
    ):
        if preview_threshold <= 0:
            return torch.cat(summed_codes, dim=-3)

        scales_in_one_clip = first_full_spatial_size_scale_index + 1
        total_scales = len(scale_schedule)

        if current_si < scales_in_one_clip:
            clip_start = 0
            clip_end = scales_in_one_clip - 1
            clip_length = scales_in_one_clip
        else:
            clip_start = scales_in_one_clip
            clip_end = total_scales - 1
            clip_length = total_scales - scales_in_one_clip

        threshold_position = clip_start + int(preview_threshold * clip_length)
        threshold_position = min(threshold_position, clip_end)

        if current_si >= threshold_position:
            return torch.cat(summed_codes, dim=-3)

        num_scales_to_preview = threshold_position - current_si

        actual_preview_scales = min(num_scales_to_preview, len(remaining_scales))

        if actual_preview_scales <= 0:
            return torch.cat(summed_codes, dim=-3)

        preview_summed_codes = [code.clone() for code in summed_codes]

        image_scale_repetition = np.array(json.loads(args.image_scale_repetition))
        video_scale_repetition = np.array(json.loads(args.video_scale_repetition))

        cum_scales = 0
        for si_past in range(current_si + 1):
            rel_si_in_one_clip = si_past % scales_in_one_clip
            if si_past < scales_in_one_clip:
                repeat_times = image_scale_repetition[rel_si_in_one_clip]
            else:
                repeat_times = video_scale_repetition[rel_si_in_one_clip]
            cum_scales += min(repeat_times, args.max_repeat_times)

        first_preview_si = current_si + 1

        if scale_schedule[current_si][-2:] == scale_schedule[-1][-2:]:
            if self.other_args.noise_input:
                preview_summed_codes.append(
                    torch.randn(
                        (B, preview_summed_codes[-1].shape[1], *vae_scale_schedule[first_preview_si]),
                        device=preview_summed_codes[-1].device,
                        dtype=preview_summed_codes[-1].dtype
                    )
                )
            else:
                preview_summed_codes.append(
                    torch.zeros(
                        (B, preview_summed_codes[-1].shape[1], *vae_scale_schedule[first_preview_si]),
                        device=preview_summed_codes[-1].device,
                        dtype=preview_summed_codes[-1].dtype
                    )
                )
            preview_last_stage = preview_summed_codes[-1]
        else:
            preview_last_stage = F.interpolate(
                preview_summed_codes[-1],
                size=vae_scale_schedule[first_preview_si],
                mode=vae.quantizer.z_interplote_down
            )

        preview_last_stage = self.embeds_codes2input(preview_last_stage, bs // B)

        for si_offset in range(actual_preview_scales):
            pn = remaining_scales[si_offset]
            si_preview = current_si + 1 + si_offset
            rel_si_in_one_clip = si_preview % scales_in_one_clip

            if si_preview < scales_in_one_clip:
                target_pn = vae_scale_schedule[first_full_spatial_size_scale_index]
                full_repeat_times = image_scale_repetition[rel_si_in_one_clip]
            else:
                target_pn = vae_scale_schedule[-1]
                full_repeat_times = video_scale_repetition[rel_si_in_one_clip]

            cfg = cfg_list[si_preview]
            infer_repeat_times = min(full_repeat_times, args.max_repeat_times)

            preview_repeat_times = 1

            for repeat_idx in range(preview_repeat_times):
                rope_cache = get_visual_rope_embeds(
                    self.rope2d_freqs_grid,
                    scale_schedule,
                    si_preview,
                    cum_scales + repeat_idx,
                    device,
                    args,
                    context_info,
                    first_full_spatial_size_scale_index
                )

                last_repetition_step = True
                for b in block_chunks:
                    preview_last_stage = b(
                        x=preview_last_stage,
                        cond_BD=cond_BD_or_gss,
                        ca_kv=ca_kv,
                        attn_bias_or_two_vector=attn_mask,
                        attn_fn=None,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=rope_cache,
                        scale_ind=si_preview,
                        context_info=context_info,
                        last_repetition_step=last_repetition_step,
                        ref_text_scale_inds=ref_text_scale_inds
                    )

                logits_BlV = self.get_logits_during_infer(
                    preview_last_stage,
                    is_semantic_scale=rel_si_in_one_clip < args.semantic_scales
                )

                if use_greedy:
                    logits_BlV = logits_BlV.mul(100.0)
                else:
                    logits_BlV = logits_BlV.mul(1 / preview_tau)

                if cfg != 1:
                    if args.use_cfg:
                        logits_BlV = cfg * logits_BlV[:B] + (1 - cfg) * logits_BlV[B:]
                    elif args.use_apg:
                        pred_cond = logits_BlV[:B]
                        pred_uncond = logits_BlV[B:]
                        pred_guided = normalized_guidance(
                            pred_cond, pred_uncond,
                            guidance_scale=cfg,
                            momentum_buffer=None,
                            eta=0,
                            norm_threshold=args.apg_norm_threshold
                        )
                        logits_BlV = pred_guided
                else:
                    logits_BlV = logits_BlV[:B]

                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)

                if use_greedy:
                    idx_Bld = logits_BlV.argmax(dim=-1)
                else:
                    probs_Bld = logits_BlV.softmax(dim=-1)
                    idx_Bld = torch.multinomial(
                        probs_Bld.view(-1, self.num_of_label_value),
                        num_samples=1,
                        replacement=True,
                        generator=rng
                    ).view(tmp_bs, -1)

                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
                idx_Bld = idx_Bld.reshape(B, pn[0], pn[1], pn[2], -1)
                if self.apply_spatial_patchify:
                    idx_Bld = idx_Bld.permute(0, 1, 4, 2, 3)
                    idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2)
                    idx_Bld = idx_Bld.permute(0, 1, 3, 4, 2)

                if self.other_args.use_two_stage_lfq:
                    if pn[1] * pn[2] >= vae.quantizer.detail_scale_min_tokens:
                        is_semantic_scale = False
                        lfq = vae.quantizer.lfq_detail
                    else:
                        is_semantic_scale = True
                        lfq = vae.quantizer.lfq_semantic

                    from infinity.schedules.infinity_elegant import interpolate
                    codes = lfq.indices_to_codes(idx_Bld, 'bit_label')
                    codes = interpolate(
                        codes,
                        size=(self.vae_embed_dim, *target_pn),
                        mode=vae.quantizer.z_interplote_up,
                        quantizer=vae.quantizer,
                        is_semantic_scale=is_semantic_scale
                    ).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)

                preview_summed_codes[-1] = F.interpolate(
                    preview_summed_codes[-1],
                    size=target_pn,
                    mode=vae.quantizer.z_interplote_up
                )
                preview_summed_codes[-1] += codes

            cum_scales += infer_repeat_times

            if si_offset < actual_preview_scales - 1:
                next_si = si_preview + 1
                if scale_schedule[si_preview][-2:] == scale_schedule[-1][-2:]:
                    if self.other_args.noise_input:
                        preview_summed_codes.append(
                            torch.randn(
                                (B, preview_summed_codes[-1].shape[1], *vae_scale_schedule[next_si]),
                                device=preview_summed_codes[-1].device,
                                dtype=preview_summed_codes[-1].dtype
                            )
                        )
                    else:
                        preview_summed_codes.append(
                            torch.zeros(
                                (B, preview_summed_codes[-1].shape[1], *vae_scale_schedule[next_si]),
                                device=preview_summed_codes[-1].device,
                                dtype=preview_summed_codes[-1].dtype
                            )
                        )
                    preview_last_stage = preview_summed_codes[-1]
                else:
                    preview_last_stage = F.interpolate(
                        preview_summed_codes[-1],
                        size=vae_scale_schedule[next_si],
                        mode=vae.quantizer.z_interplote_down
                    )

                preview_last_stage = self.embeds_codes2input(preview_last_stage, bs // B)

        preview_summed_codes = torch.cat(preview_summed_codes, dim=-3)

        return preview_summed_codes


    @torch.no_grad()
    def ar_infer_infinity_elegant(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1,
        negative_label_B_or_BLT=None,
        g_seed=None,
        cfg_list=[],
        tau_list=[],
        top_k=0,
        top_p=0.0,
        trunk_scale=1000,
        gt_leak=0,
        gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        context_info=None,
        return_summed_code_only=False,
        **kwargs,
    ):   # returns List[idx_Bl]
        from infinity.schedules.infinity_elegant import interpolate
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_cfg + args.use_apg == 1
        device = label_B_or_BLT[0].device
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = label_B_or_BLT[-1] # self.text_maxlen # kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        prefix_tokens, lens = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT, vae_scale_schedule, text_token_only=False, text_maxlen_this_iter=text_maxlen_this_iter)
        bs = prefix_tokens.shape[0]
        ca_kv, cond_BD_or_gss, attn_mask = None, None, None
        ret, idx_Bl_list = [], []  # current length, list of reconstructed images
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        image_scale_repetition = np.array(json.loads(args.image_scale_repetition))
        video_scale_repetition = np.array(json.loads(args.video_scale_repetition))
        scales_in_one_clip = first_full_spatial_size_scale_index + 1
        assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
        assert len(image_scale_repetition) == scales_in_one_clip, f'{len(image_scale_repetition)} != {scales_in_one_clip}'
        total_steps = image_scale_repetition.sum() + video_scale_repetition.sum() * (len(scale_schedule)//len(video_scale_repetition)-1) + 1 # +1 is prefix text token forward step
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks

        noise_shape = vae_scale_schedule[0]
        if self.other_args.noise_input:
            noise = torch.randn((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)
        else:
            noise = torch.zeros((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)

        summed_codes = [noise[0:1]]
        sos_token = self.embeds_codes2input(noise, bs//1)

        rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:text_maxlen_this_iter]
        last_stage = prefix_tokens
        pbar.update(1)
        for block_idx, b in enumerate(block_chunks):
            last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind='t0', context_info=context_info, last_repetition_step=True)

        ref_text_scale_inds = ['t0']
        last_stage = sos_token
        cum_scales = 0
        prev_last_stage = None
        prev_summed_codes = None

        for si, pn in enumerate(scale_schedule):
            rel_si_in_one_clip = si % scales_in_one_clip
            if si < scales_in_one_clip: # image
                repeat_times = image_scale_repetition[si%scales_in_one_clip]
                target_pn = vae_scale_schedule[first_full_spatial_size_scale_index]
            else:
                repeat_times = video_scale_repetition[si%scales_in_one_clip]
                target_pn = vae_scale_schedule[-1]
            cfg = cfg_list[si]
            infer_repeat_times = min(repeat_times, args.max_repeat_times)

            for repeat_idx in range(infer_repeat_times):
                # print(f'real scale ind is : {cum_scales+repeat_idx}')

                prev_last_stage = last_stage.clone()
                prev_summed_codes = [code.clone() for code in summed_codes]

                rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule, si, cum_scales+repeat_idx, device, args, context_info, first_full_spatial_size_scale_index)
                pbar.update(1)

                last_repetition_step = (repeat_idx == (infer_repeat_times-1))
                for block_idx, b in enumerate(block_chunks):
                    last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds)

                logits_BlV = self.get_logits_during_infer(last_stage, is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(1/tau_list[si])

                if cfg != 1:
                    # print(f'add cfg on add_cfg_on_logits')
                    if args.use_cfg:
                        logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                    elif args.use_apg:
                        pred_cond = logits_BlV[:B]
                        pred_uncond = logits_BlV[B:]
                        pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg, momentum_buffer=None, eta=0, norm_threshold=args.apg_norm_threshold)
                        # pred_guided = cfg * pred_cond + (1-cfg) * pred_uncond
                        logits_BlV = pred_guided
                else:
                    logits_BlV = logits_BlV[:B]

                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)

                probs_Bld = logits_BlV.softmax(dim=-1) # [B, thwd or thw4d, 2]
                idx_Bld = torch.multinomial(probs_Bld.view(-1, self.num_of_label_value), num_samples=1, replacement=True, generator=rng).view(tmp_bs, -1) # [B, thwd or thw4d]
                probs_Bld = torch.gather(probs_Bld, dim=2, index=idx_Bld.unsqueeze(-1)).squeeze(-1)

                def Bld2Bthwd(item):
                    item = item.reshape(tmp_bs, tmp_seq_len, -1) # [B, thw, d or 4d]
                    item = item.reshape(B, pn[0], pn[1], pn[2], -1) # shape: [B, t, h, w, d] or [B, t, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        item = item.permute(0,1,4,2,3) # [B, t, 4d, h, w]
                        item = torch.nn.functional.pixel_shuffle(item, 2) # [B, t, d, 2h, 2w]
                        item = item.permute(0,1,3,4,2) # [B, t, 2h, 2w, d]
                    return item

                idx_Bld = Bld2Bthwd(idx_Bld)
                probs_Bld = Bld2Bthwd(probs_Bld)
                # print(f'{si=} {repeat_idx=} idx_Bld.shape={idx_Bld.shape}')

                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[cum_scales+repeat_idx]
                # idx_Bld [B, t, h, w, d] or [B, t, 2h, 2w, d]

                if self.other_args.use_two_stage_lfq:
                    if pn[1] * pn[2] >= vae.quantizer.detail_scale_min_tokens:
                        is_semantic_scale = False
                        lfq = vae.quantizer.lfq_detail
                    else:
                        is_semantic_scale = True
                        lfq = vae.quantizer.lfq_semantic
                    codes = lfq.indices_to_codes(idx_Bld, 'bit_label')
                    codes = interpolate(codes, size=(self.vae_embed_dim, *target_pn), mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)

                summed_codes[-1] = F.interpolate(summed_codes[-1], size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] += codes
                if repeat_idx < repeat_times - 1:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down)
                    last_stage = self.embeds_codes2input(last_stage, bs//B)

            cum_scales += repeat_times

            should_reflect = False
            if hasattr(args, 'enable_self_reflection') and args.enable_self_reflection:
                total_scales = len(scale_schedule)
                start_scale = max(int(total_scales * args.reflection_start_scale), 1)
                end_scale = min(int(total_scales * args.reflection_end_scale), len(scale_schedule)-2)
                if (
                    si >= start_scale and
                    si <= end_scale and
                    (si - start_scale) % args.reflection_interval == 0
                ):
                    should_reflect = True

            if should_reflect:
                remaining_scale_indices = list(range(si + 1, len(scale_schedule)))
                remaining_scales = [scale_schedule[i] for i in remaining_scale_indices]

                preview_summed_codes = self.fast_forward_preview(
                    current_si=si,
                    last_stage=last_stage,
                    summed_codes=summed_codes,
                    scale_schedule=scale_schedule,
                    vae_scale_schedule=vae_scale_schedule,
                    remaining_scales=remaining_scales,
                    vae=vae,
                    cfg_list=cfg_list,
                    tau_list=tau_list,
                    B=B,
                    bs=bs,
                    cond_BD_or_gss=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_mask=attn_mask,
                    block_chunks=block_chunks,
                    device=device,
                    args=args,
                    context_info=context_info,
                    first_full_spatial_size_scale_index=first_full_spatial_size_scale_index,
                    get_visual_rope_embeds=get_visual_rope_embeds,
                    rng=rng,
                    ref_text_scale_inds=ref_text_scale_inds,
                    use_greedy=False,
                    preview_tau=0.1,
                )
                preview_img = self.summed_codes2images(vae, preview_summed_codes)[0]

                preview_img_list = []
                if len(preview_img.shape) == 3:
                    preview_img = preview_img.unsqueeze(0)
                preview_img_list.append(preview_img)
                preview = torch.cat(preview_img_list, 2).cpu().numpy()

                random_uuid = uuid.uuid4().hex
                video_path = os.path.join(os.getcwd(), f"temp_{random_uuid}.mp4")
                media_type = 'video'
                if len(preview) == 1:
                    video_path = video_path.replace('.mp4', '.jpg')
                    media_type = 'image'
                save_video(preview, fps=args.fps, save_filepath=video_path)

                feedback = quick_evaluate_media(video_path, args.prompt, media_type=media_type) # TODO: is match?
                diagnosis = feedback["diagnosis"]
                enhanced_prompts = feedback["enhanced_prompt"]
                negative_prompts = feedback["negative_prompt"] # TODO: take too much time
                print(f"获得评估结果: {diagnosis}")
                print(f"获得增强提示词: {enhanced_prompts}")
                print(f"获得负面提示词: {negative_prompts}")

                os.remove(video_path)

                # encode prompt
                label_B_or_BLT = encode_prompt(args.text_encoder_ckpt, args.text_tokenizer, args.text_encoder, enhanced_prompts)
                negative_label_B_or_BLT = encode_prompt(args.text_encoder_ckpt, args.text_tokenizer, args.text_encoder, negative_prompts)
                text_maxlen_this_iter=max(label_B_or_BLT[-1], negative_label_B_or_BLT[-1])
                enhanced_prefix_tokens, _ = self.prepare_text_conditions(
                    label_B_or_BLT,
                    cfg_list,
                    B,
                    negative_label_B_or_BLT,
                    vae_scale_schedule,
                    text_token_only=False,
                    text_maxlen_this_iter=text_maxlen_this_iter
                )
                enhanced_text_scale_ind = f't{si+1}'
                enhanced_rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:text_maxlen_this_iter]

                enhanced_last_stage = enhanced_prefix_tokens
                for block_idx, b in enumerate(block_chunks):
                    enhanced_last_stage = b(
                        x=enhanced_last_stage,
                        cond_BD=cond_BD_or_gss,
                        ca_kv=ca_kv,
                        attn_bias_or_two_vector=attn_mask,
                        attn_fn=None,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=enhanced_rope_cache,
                        scale_ind=enhanced_text_scale_ind,
                        context_info=context_info,
                        last_repetition_step=True
                    )


                if args.use_only_enhanced_prompt:
                    ref_text_scale_inds = [enhanced_text_scale_ind]
                else:
                    ref_text_scale_inds.append(enhanced_text_scale_ind)
                    if len(ref_text_scale_inds) > 2:
                        old_text_scale_ind = ref_text_scale_inds.pop(1)
                        for block_idx, block in enumerate(self.unregistered_blocks):
                            if old_text_scale_ind in block.attn.cached_k:
                                del block.attn.cached_k[old_text_scale_ind]
                                del block.attn.cached_v[old_text_scale_ind]
                        print(f"已清理旧文本缓存: {old_text_scale_ind}")

                print(f"已缓存增强文本特征，scale_ind={enhanced_text_scale_ind}")
                print(f"当前引用的文本scale: {ref_text_scale_inds}")

                last_stage = prev_last_stage.clone()
                summed_codes = [code.clone() for code in prev_summed_codes]

                prev_last_stage = last_stage.clone()
                prev_summed_codes = [code.clone() for code in summed_codes]

                rope_cache = get_visual_rope_embeds(
                    self.rope2d_freqs_grid, scale_schedule, si,
                    cum_scales-1, device, args, context_info,
                    first_full_spatial_size_scale_index
                )

                for block_idx, b in enumerate(block_chunks):
                    last_stage = b(
                        x=last_stage,
                        cond_BD=cond_BD_or_gss,
                        ca_kv=ca_kv,
                        attn_bias_or_two_vector=attn_mask,
                        attn_fn=None,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=rope_cache,
                        scale_ind=si,
                        context_info=context_info,
                        last_repetition_step=True,
                        ref_text_scale_inds=ref_text_scale_inds
                    )

                logits_BlV = self.get_logits_during_infer(
                    last_stage,
                    is_semantic_scale=rel_si_in_one_clip < args.semantic_scales
                ).mul(1/tau_list[si])

                if cfg != 1:
                    if args.use_cfg:
                        logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                    elif args.use_apg:
                        pred_cond = logits_BlV[:B]
                        pred_uncond = logits_BlV[B:]
                        pred_guided = normalized_guidance(
                            pred_cond, pred_uncond, guidance_scale=cfg,
                            momentum_buffer=None, eta=0,
                            norm_threshold=args.apg_norm_threshold
                        )
                        logits_BlV = pred_guided
                else:
                    logits_BlV = logits_BlV[:B]

                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)

                probs_Bld = logits_BlV.softmax(dim=-1)
                idx_Bld = torch.multinomial(
                    probs_Bld.view(-1, self.num_of_label_value),
                    num_samples=1, replacement=True, generator=rng
                ).view(tmp_bs, -1)

                idx_Bld = Bld2Bthwd(idx_Bld)

                if self.other_args.use_two_stage_lfq:
                    if pn[1] * pn[2] >= vae.quantizer.detail_scale_min_tokens:
                        is_semantic_scale = False
                        lfq = vae.quantizer.lfq_detail
                    else:
                        is_semantic_scale = True
                        lfq = vae.quantizer.lfq_semantic
                    codes = lfq.indices_to_codes(idx_Bld, 'bit_label')
                    codes = interpolate(
                        codes, size=(self.vae_embed_dim, *target_pn),
                        mode=vae.quantizer.z_interplote_up,
                        quantizer=vae.quantizer,
                        is_semantic_scale=is_semantic_scale
                    ).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)

                summed_codes[-1] = F.interpolate(
                    summed_codes[-1], size=target_pn, mode=vae.quantizer.z_interplote_up
                )
                summed_codes[-1] += codes

            if si < len(scale_schedule)-1:
                if scale_schedule[si][-2:] == scale_schedule[-1][-2:]:
                    if self.other_args.noise_input:
                        summed_codes.append(torch.randn((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    else:
                        summed_codes.append(torch.zeros((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    last_stage = summed_codes[-1]
                else:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_down)
                last_stage = self.embeds_codes2input(last_stage, bs//B)

        summed_codes = torch.cat(summed_codes, dim=-3)
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        if return_summed_code_only:
            return summed_codes
        else:
            if low_vram_mode: vae.to('cuda')
            img = self.summed_codes2images(vae, summed_codes)
            return idx_Bl_list, img


    @torch.no_grad()
    def ar_infer_infinity_star_interact(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        context_info=None,
        return_summed_code_only=False,
        mode='',
        former_clip_features=None,
        first_frame_features=None,
        semantic_scale_ind = 7,
        detail_frame_inds = [18,19],
        **kwargs,
    ):   # returns List[idx_Bl]
        from infinity.schedules.infinity_star_interact import interpolate
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_apg + args.use_cfg == 1
        device = label_B_or_BLT[0].device
        if g_seed is None:
            rng = None
        else:
            self.rng = torch.Generator(device=device)
            self.rng.manual_seed(g_seed)
            rng = self.rng

        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = label_B_or_BLT[-1] # self.text_maxlen # kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        prefix_tokens, _ = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT, vae_scale_schedule, text_token_only=False, text_maxlen_this_iter=text_maxlen_this_iter)
        bs = prefix_tokens.shape[0]

        ca_kv, cond_BD_or_gss, attn_mask = None, None, None
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        image_scale_repetition = np.array(json.loads(args.image_scale_repetition))
        video_scale_repetition = np.array(json.loads(args.video_scale_repetition))
        scales_in_one_clip = first_full_spatial_size_scale_index + 1
        assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
        assert len(image_scale_repetition) == scales_in_one_clip, f'{len(image_scale_repetition)} != {scales_in_one_clip}'
        total_steps = image_scale_repetition.sum() + video_scale_repetition.sum() * (len(scale_schedule)//len(video_scale_repetition)-1) + 1 # +1 is prefix text token forward step
        if mode == 'second_v_clip':
            total_steps += 2
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks

        noise_shape = vae_scale_schedule[0]
        if self.other_args.noise_input:
            noise = torch.randn((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)
        else:
            noise = torch.zeros((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)

        summed_codes = [noise[0:1]]
        sos_token = self.embeds_codes2input(noise, bs//1)
        # text tokens forward
        rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:text_maxlen_this_iter]
        last_stage = prefix_tokens
        for block_idx, b in enumerate(block_chunks):
            last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=f't0', context_info=context_info, last_repetition_step=True)
        pbar.update(1)

        ref_text_scale_inds = ['t0']

        # visual condition forward
        if mode == 'second_v_clip':
            assert former_clip_features.shape[-3] == 21
            former_clip_features = former_clip_features[:,:,1:]
            last_stage = F.interpolate(former_clip_features, size=(20, *vae_scale_schedule[semantic_scale_ind][-2:]), mode=vae.quantizer.z_interplote_down)
            rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule[-1], last_stage.shape[-3:], list(range(1, 21)), 800, device)
            last_stage = self.embeds_codes2input(last_stage, bs//B)
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=f'semantic_condition', context_info=context_info, last_repetition_step=True)
            pbar.update(1)

            last_stage = torch.cat([first_frame_features, former_clip_features[:,:,detail_frame_inds]], dim=2)
            rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule[-1], last_stage.shape[-3:], [0]+[item+1 for item in detail_frame_inds], 801, device)
            last_stage = self.embeds_codes2input(last_stage, bs//B)
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=f'detail_condition', context_info=context_info, last_repetition_step=True)
            pbar.update(1)

            ref_text_scale_inds.extend(['semantic_condition', 'detail_condition'])

        # visual tokens forward
        last_stage = sos_token
        cum_scales = 0
        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            rel_si_in_one_clip = si % scales_in_one_clip
            if si < scales_in_one_clip: # image
                repeat_times = image_scale_repetition[rel_si_in_one_clip]
                target_pn = vae_scale_schedule[first_full_spatial_size_scale_index]
            else:
                repeat_times = video_scale_repetition[rel_si_in_one_clip]
                target_pn = vae_scale_schedule[-1]
            cfg = cfg_list[si]
            infer_repeat_times = min(repeat_times, args.max_repeat_times)
            for repeat_idx in range(infer_repeat_times):
                frame_ss, frame_ee = context_info[si]['frame_ss'], context_info[si]['frame_ee']
                rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule[-1], scale_schedule[si], list(range(frame_ss, frame_ee)), cum_scales+repeat_idx, device)
                last_repetition_step = (repeat_idx == (infer_repeat_times-1))
                for block_idx, b in enumerate(block_chunks):
                    last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds)
                logits_BlV = self.get_logits_during_infer(last_stage, is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(1/tau_list[si])
                if cfg != 1:
                    # print(f'add cfg on add_cfg_on_logits')
                    if args.use_cfg:
                        logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                    elif args.use_apg:
                        pred_cond = logits_BlV[:B]
                        pred_uncond = logits_BlV[B:]
                        pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg, momentum_buffer=None, eta=0, norm_threshold=args.apg_norm_threshold)
                        # pred_guided = cfg * pred_cond + (1-cfg) * pred_uncond
                        logits_BlV = pred_guided
                else:
                    logits_BlV = logits_BlV[:B]

                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)
                probs_Bld = logits_BlV.softmax(dim=-1) # [B, thwd or thw4d, 2]
                idx_Bld = torch.multinomial(probs_Bld.view(-1, self.num_of_label_value), num_samples=1, replacement=True, generator=rng).view(tmp_bs, -1) # [B, thwd or thw4d]
                probs_Bld = torch.gather(probs_Bld, dim=2, index=idx_Bld.unsqueeze(-1)).squeeze(-1)

                def Bld2Bthwd(item):
                    item = item.reshape(tmp_bs, tmp_seq_len, -1) # [B, thw, d or 4d]
                    item = item.reshape(B, pn[0], pn[1], pn[2], -1) # shape: [B, t, h, w, d] or [B, t, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        item = item.permute(0,1,4,2,3) # [B, t, 4d, h, w]
                        item = torch.nn.functional.pixel_shuffle(item, 2) # [B, t, d, 2h, 2w]
                        item = item.permute(0,1,3,4,2) # [B, t, 2h, 2w, d]
                    return item

                idx_Bld = Bld2Bthwd(idx_Bld)
                probs_Bld = Bld2Bthwd(probs_Bld)

                if si < gt_leak:
                    acc = (idx_Bld==gt_ls_Bl[cum_scales+repeat_idx]).float().mean() * 100.
                    idx_Bld = gt_ls_Bl[cum_scales+repeat_idx]
                    print(f'{si=} {repeat_idx=} idx_Bld.shape={idx_Bld.shape} {acc=}%')

                # idx_Bld [B, t, h, w, d] or [B, t, 2h, 2w, d]
                if self.other_args.use_two_stage_lfq:
                    if si >= args.semantic_scales:
                        is_semantic_scale = False
                        lfq = vae.quantizer.lfq_detail
                    else:
                        is_semantic_scale = True
                        lfq = vae.quantizer.lfq_semantic
                    codes = lfq.indices_to_codes(idx_Bld, 'bit_label')
                    codes = interpolate(codes, size=(self.vae_embed_dim, *target_pn), mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] = F.interpolate(summed_codes[-1], size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] += codes
                if repeat_idx < repeat_times - 1:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down)
                    last_stage = self.embeds_codes2input(last_stage, bs//B)
                pbar.update(1)
            cum_scales += repeat_times
            if si < len(scale_schedule)-1:
                if scale_schedule[si][-2:] == scale_schedule[-1][-2:]:
                    if self.other_args.noise_input:
                        summed_codes.append(torch.randn((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    else:
                        summed_codes.append(torch.zeros((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    last_stage = summed_codes[-1]
                else:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_down)
                last_stage = self.embeds_codes2input(last_stage, bs//B)
        summed_codes = torch.cat(summed_codes, dim=-3)
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        if mode == 'second_v_clip':
            this_clip_frames = summed_codes.shape[2] * 4
            summed_codes = torch.cat([former_clip_features, summed_codes], dim=-3)
            img = self.summed_codes2images(vae, summed_codes) # [bs, t, h, w, 3]
            img = img[:,-this_clip_frames:]
            summed_codes = summed_codes[:,:,-21:]
            assert summed_codes.shape[2] == 21, f'wrong shape: {summed_codes.shape=}'
        else:
            img = self.summed_codes2images(vae, summed_codes)

        if low_vram_mode: vae.to('cuda')
        return summed_codes, img

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        returns_vemb=0,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        **kwargs,
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_cfg + args.use_apg == 1
        device = label_B_or_BLT[0].device
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = self.text_maxlen
        last_stage, lens, _ = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT, args.input_noise, vae_scale_schedule)
        bs = last_stage.shape[0]
        ca_kv, cond_BD_or_gss = None, None
        ret, idx_Bl_list = [], []  # current length, list of reconstructed images
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            visual_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule, si, device, args)
            if si == 0:
                rope_cache = torch.cat([self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:text_maxlen_this_iter], visual_rope_cache], dim=4)
            else:
                rope_cache = visual_rope_cache
            attn_mask = torch.ones((last_stage.shape[0], 1, last_stage.shape[1], text_maxlen_this_iter+np.array(pn).prod()), device=last_stage.device).bool() # [bs, q_heads, q_len, all_k_len], here set q_heads=1 for broadcasting
            assert len(attn_mask) == len(lens)
            for tmp_i, le in enumerate(lens):
                attn_mask[tmp_i, :, :, le:text_maxlen_this_iter] = False
                if si == 0:
                    attn_mask[tmp_i, :, :text_maxlen_this_iter, text_maxlen_this_iter:] = False
            cfg = cfg_list[si]
            if si >= trunk_scale:
                break
            for block_idx, b in enumerate(self.block_chunks):
                for m in b.module:
                    last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si)
            if si == 0:
                last_stage = last_stage[:, text_maxlen_this_iter:]
            # import pdb; pdb.set_trace()
            if cfg != 1:
                # print(f'add cfg on add_cfg_on_logits')
                logits_BlV = self.get_logits(last_stage).mul(1/tau_list[si])
                if args.use_cfg:
                    logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                elif args.use_apg:
                    pred_cond = logits_BlV[:B]
                    pred_uncond = logits_BlV[B:]
                    pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg, momentum_buffer=None, eta=0, norm_threshold=10)
                    # pred_guided = cfg * pred_cond + (1-cfg) * pred_uncond
                    logits_BlV = pred_guided
            else:
                logits_BlV = self.get_logits(last_stage[:B]).mul(1/tau_list[si])
            if self.num_of_label_value == 1:
                idx_Bld = logits_BlV
            elif self.num_of_label_value > 1:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            elif self.num_of_label_value == 0:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            assert returns_vemb
            if si < gt_leak:
                idx_Bld = gt_ls_Bl[si]
            else:
                idx_Bld = idx_Bld.reshape(B, pn[0], pn[1], pn[2], -1) # shape: [B, t, h, w, d] or [B, t, h, w, 4d]
                if self.apply_spatial_patchify: # unpatchify operation
                    idx_Bld = idx_Bld.permute(0,1,4,2,3) # [B, t, 4d, h, w]
                    idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, t, d, 2h, 2w]
                    idx_Bld = idx_Bld.permute(0,1,3,4,2) # [B, t, 2h, 2w, d]
                # idx_Bld [B, t, h, w, d] or [B, t, 2h, 2w, d]

            # idx_Bld_list.append(idx_Bld)
            if self.num_of_label_value == 1:
                if si < gt_leak:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, t, h, w] or [B, d, t, 2h, 2w]
                else:
                    codes = idx_Bld.permute(0,4,1,2,3)
            else:
                codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, t, h, w] or [B, d, t, 2h, 2w]
            if vae_scale_schedule[si] != vae_scale_schedule[-1]:
                codes = F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            summed_codes += codes
            if si < len(scale_schedule)-1:
                last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_down) # [B, d, t, h, w] or [B, d, t, 2h, 2w]
                if self.apply_spatial_patchify: # patchify operation
                    last_stage = last_stage.permute(0,2,1,3,4) # [B, t, d, 2h, 2w]
                    last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, t, 4d, h, w]
                    last_stage = last_stage.permute(0,2,1,3,4) # [B, 4d, t, h, w]
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, t*h*w] or [B, 4d, t*h*w]
                last_stage = torch.permute(last_stage, [0,2,1]) # [B, t*h*w, d] or [B, t*h*w, 4d]
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        if low_vram_mode: vae.to('cuda')
        img = self.summed_codes2images(vae, summed_codes)
        return ret, idx_Bl_list, img

    def summed_codes2images(self, vae, summed_codes):
        t1 = time.time()

        img = vae.decode(summed_codes, slice=True)
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        img = img.permute(0,2,3,4,1) # [bs, 3, t, h, w] -> [bs, t, h, w, 3]
        img = img.mul_(255).to(torch.uint8).flip(dims=(4,))

        # smooth the image & video
        if img.shape[1] >= 2:
            img[:, 0:1, :, :, :] = img[:, 1:2, :, :, :]

        print(f'Decode takes {time.time()-t1:.1f}s')
        return img

    @for_visualize
    def vis_key_params(self, ep):
        return

    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]

        for buf_name in ('lvl_1L', 'attn_bias_for_masking', 'Infinity_visible_kvlen', 'Infinity_invisible_qlen'):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)

        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

    def special_init(self):
        if self.arch == 'qwen':
            std = 0.02
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
        else:
            raise ValueError(f'Unknown arch {self.arch}')

    def extra_repr(self):
        return f''

    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sampling_with_top_k_top_p_also_inplace_modifying_probs_(probs_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = probs_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = probs_BlV < probs_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        probs_BlV.masked_fill_(idx_to_remove, 0)
    if top_p > 0:
        sorted_probs, sorted_idx = probs_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_probs.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        probs_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 0)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    probs_BlV = probs_BlV / probs_BlV.sum(-1, keepdims=True)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs_BlV.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def get_params_num(d, w, mlp):
    m = round(mlp * w / 256) * 256
    s = d * (w**2 * 8 + w*m * 2)    # sa+ca, mlp
    s += w**2 * 6       # saln
    s += 4096 * w       # pred
    s += 32 * w         # we

    Ct5 = 4096
    s += Ct5*w * 4      # T5 attn pool
    s += Ct5*w + w*w    # T5 mlp
    return f'{s/1e9:.2f}B'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

@register_model
def infinity_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa2b(depth=28, block_chunks=7, embed_dim=2560, num_heads=2560//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa8b(depth=42, block_chunks=7, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa14b(depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, drop_path_rate=0.1, mlp_ratio=3.4, **kwargs):
    return Infinity(
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )
    # (depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, num_key_value_heads=5120//128//4, drop_path_rate=0, **kwargs)

@register_model
def infinity_sa12b(depth=60, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa16b(depth=42, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_v2b(depth=32, embed_dim=2016, num_heads=2016//126, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_8b(depth=40, block_chunks=1, embed_dim=3584, num_heads=3584//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_qwen7b(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096//128, num_key_value_heads=4096//128//4, mlp_ratio=12288/4096, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen8b(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096//128, num_key_value_heads=4096//128//4, mlp_ratio=4, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen_wide14b(depth=36, block_chunks=6, embed_dim=5632, num_heads=5632//128, num_key_value_heads=5632//128//4, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen13bMHA(depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, num_key_value_heads=5120//128, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=True,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen2_2b(depth=28, block_chunks=7, embed_dim=2304, num_heads=2304//128, num_key_value_heads=2304//128, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen0b(depth=4, block_chunks=2, embed_dim=512, num_heads=512//128, num_key_value_heads=512//128, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen2_30b(depth=54, block_chunks=27, embed_dim=6144, num_heads=6144//128, num_key_value_heads=6144//128//4, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4, #mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen14b(depth=48, block_chunks=24, embed_dim=4608, num_heads=4608//128, num_key_value_heads=4608//128//4, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_20b(depth=58, embed_dim=4608, num_heads=4608//128, drop_path_rate=0.25, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

# model configuration for scaling Infinity transformer
@register_model
def infinity_layer12(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, **kwargs):
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer16(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, **kwargs):
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer24(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, **kwargs):
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer32(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, **kwargs):
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer40(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, **kwargs):
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer48(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, **kwargs):
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
