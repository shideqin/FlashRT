"""
OmniVoice FlashRT Engine — flashcli bundle module.

The quality-safe profile runs every CFG step through an exact BF16 path.
Lower ``cfg_ratio`` values opt into a faster BF16/FP4 hybrid with a possible
token and audio-quality tradeoff.

Architecture:
  FlashRTLlm       — FP4 W4A4 forward (CUDA Graph + FA2 + fused kernels)
  FlashRTLlmBF16   — exact BF16 forward (CUDA Graph + SDPA + fused kernels)
  inject()         — Patch OmniVoice model for FlashRT acceleration
  free_encoder()   — Release encoder weights to save VRAM
"""
from __future__ import annotations
from collections import OrderedDict
import logging, math, numpy as np, torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
BF16 = torch.bfloat16

# ── Qwen3-1.5B model constants ──
D = 1024; L = 28; NH = 16; NKV = 8; HD = 128; FFN = 3072; EPS = 1e-6
THETA = 1_000_000.0; NQK = NH * HD; KVD = NKV * HD
QKVD = NQK + 2 * KVD; GUD = 2 * FFN; MAX_S = 2048
MASKGIT_CUDA_SELECT_MAX_ELEMENTS = 1024
_FADE_RAMP_CACHE = {}


def _remove_silence_numpy_exact(
        audio, sampling_rate, mid_sil=500,
        lead_sil=100, trail_sil=100):
    """Vectorized equivalent of OmniVoice's int16/pydub silence pass."""
    samples=(audio*32768.0).clip(-32768,32767).astype(
        np.int16,copy=False)
    channels,total_frames=samples.shape
    threshold_next=104

    def length_ms(frames):
        return round(1000*frames/sampling_rate)

    def frame_at(ms):
        return int(ms*(sampling_rate/1000.0))

    def quiet_windows(data, starts_ms, duration_ms):
        if not starts_ms:
            return np.empty(0,dtype=np.bool_)
        squared=data.astype(np.int64)**2
        per_frame=squared.sum(axis=0,dtype=np.int64)
        prefix=np.empty(per_frame.size+1,dtype=np.int64)
        prefix[0]=0
        np.cumsum(per_frame,dtype=np.int64,out=prefix[1:])
        starts=np.fromiter(
            (frame_at(ms) for ms in starts_ms),
            dtype=np.int64,count=len(starts_ms))
        requested_ends=np.fromiter(
            (frame_at(ms+duration_ms) for ms in starts_ms),
            dtype=np.int64,count=len(starts_ms))
        ends=np.minimum(requested_ends,data.shape[-1])
        sums=prefix[ends]-prefix[np.minimum(starts,data.shape[-1])]
        counts=(requested_ends-starts)*channels
        # audioop.rms truncates sqrt(mean). rms <= 103 exactly when the
        # unrounded mean is below 104**2.
        return sums<(threshold_next*threshold_next)*counts

    def slice_ms(data, start_ms, end_ms):
        start=max(0,frame_at(start_ms))
        end=max(start,frame_at(end_ms))
        available=data[:,min(start,data.shape[-1]):min(end,data.shape[-1])]
        missing=end-start-available.shape[-1]
        if missing>0:
            available=np.pad(available,((0,0),(0,missing)))
        return available

    if mid_sil>0:
        segment_ms=length_ms(total_frames)
        silent_ranges=[]
        if segment_ms>=mid_sil:
            last=segment_ms-mid_sil
            starts=list(range(0,last+1,10))
            if last%10:
                starts.append(last)
            silent_starts=[
                start for start,quiet in zip(
                    starts,quiet_windows(samples,starts,mid_sil))
                if quiet]
            if silent_starts:
                range_start=previous=silent_starts[0]
                for start in silent_starts[1:]:
                    continuous=start==previous+10
                    has_gap=start>previous+mid_sil
                    if not continuous and has_gap:
                        silent_ranges.append(
                            [range_start,previous+mid_sil])
                        range_start=start
                    previous=start
                silent_ranges.append(
                    [range_start,previous+mid_sil])
        if silent_ranges:
            nonsilent=[]
            previous_end=0
            for start,end in silent_ranges:
                nonsilent.append([previous_end,start])
                previous_end=end
            if silent_ranges[-1][1]!=segment_ms:
                nonsilent.append([previous_end,segment_ms])
            if nonsilent and nonsilent[0]==[0,0]:
                nonsilent.pop(0)
            ranges=[
                [start-mid_sil,end+mid_sil]
                for start,end in nonsilent]
            for left,right in zip(ranges,ranges[1:]):
                if right[0]<left[1]:
                    boundary=(left[1]+right[0])//2
                    left[1]=boundary
                    right[0]=boundary
            pieces=[
                slice_ms(
                    samples,max(start,0),min(end,segment_ms))
                for start,end in ranges]
            samples=(
                np.concatenate(pieces,axis=-1)
                if pieces else np.empty((1,0),dtype=np.int16))

    def trim_leading_ms(data):
        segment_ms=length_ms(data.shape[-1])
        starts=list(range(0,segment_ms,10))
        quiet=quiet_windows(data,starts,10)
        trim=0
        for is_quiet in quiet:
            if not is_quiet:
                break
            trim+=10
        return min(trim,segment_ms)

    start=max(0,trim_leading_ms(samples)-lead_sil)
    samples=slice_ms(samples,start,length_ms(samples.shape[-1]))
    samples=samples[:,::-1]
    start=max(0,trim_leading_ms(samples)-trail_sil)
    samples=slice_ms(samples,start,length_ms(samples.shape[-1]))
    samples=samples[:,::-1]
    return samples.astype(np.float32)/32768.0


def _fade_and_pad_numpy_exact(
        audio,pad_duration=0.1,fade_duration=0.1,
        sample_rate=24000,volume_op=None):
    if audio.shape[-1]==0:
        return audio
    fade_samples=int(fade_duration*sample_rate)
    pad_samples=int(pad_duration*sample_rate)
    output=np.zeros(
        (audio.shape[0],audio.shape[-1]+2*pad_samples),
        dtype=audio.dtype)
    processed=output[
        :,pad_samples:pad_samples+audio.shape[-1]]
    processed[...] = audio
    if volume_op is not None:
        operation,value=volume_op
        if operation=="ref":
            np.multiply(processed,value,out=processed)
            np.divide(processed,0.1,out=processed)
        else:
            np.divide(processed,value,out=processed)
            np.multiply(processed,0.5,out=processed)
    if fade_samples>0:
        k=min(fade_samples,processed.shape[-1]//2)
        if k>0:
            ramps=_FADE_RAMP_CACHE.get(k)
            if ramps is None:
                ramps=(
                    np.linspace(0,1,k,dtype=np.float32)[None,:],
                    np.linspace(1,0,k,dtype=np.float32)[None,:])
                _FADE_RAMP_CACHE[k]=ramps
            processed[...,:k]*=ramps[0]
            processed[...,-k:]*=ramps[1]
    return output

# ── FlashRT kernel detection (supports flashcli deployment + local dev) ──
try:
    from flash_rt import flash_rt_kernels as _fvk
except ImportError:
    try:
        import flash_rt_kernels as _fvk
    except ImportError:
        _fvk = None

try:
    from flash_rt import flash_rt_omnivoice as _fvo
except ImportError:
    try:
        import flash_rt_omnivoice as _fvo
    except ImportError:
        _fvo = None

_has_cfg_kernel = _fvo is not None and hasattr(_fvo, 'omnivoice_cfg_logsoftmax_bf16')


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def _swizzled_sf_bytes(rows, cols):
    assert cols % 16 == 0
    n_blocks = cols // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 128 * 64

def _pick_gemm_variant(N, K):
    return 'pingpong' if N >= 4096 else 'default'

def _call_fp4_gemm(fvk, A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, variant, stream):
    if variant == 'pingpong':
        fvk.fp4_w4a16_gemm_sm120_bf16out_pingpong(A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream)
    else:
        fvk.fp4_w4a16_gemm_sm120_bf16out(A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream)


# ═══════════════════════════════════════════════════════════════
# FlashRTLlm — FP4 W4A4 forward engine
# ═══════════════════════════════════════════════════════════════

class FlashRTLlm:
    """FlashRT FP4 LLM — NVFP4 W4A4 GEMM + fused QK norm+rope + FA2 + CUDA Graph."""

    def __init__(self, qm, dev="cuda:0"):
        self.device = dev
        layers = qm.layers
        from flash_rt import flash_rt_kernels as fvk
        from flash_rt import flash_rt_omnivoice as fvo
        from flash_rt import flash_rt_fa2 as fa2
        self.fvk = fvk; self.fvo = fvo; self.fa2 = fa2

        self.WL_bf16 = []
        for i in range(L):
            ly = layers[i]
            self.WL_bf16.append({
                'qkv': torch.cat([ly.self_attn.q_proj.weight.data,
                                  ly.self_attn.k_proj.weight.data,
                                  ly.self_attn.v_proj.weight.data], 0).contiguous(),
                'gu': torch.cat([ly.mlp.gate_proj.weight.data,
                                  ly.mlp.up_proj.weight.data], 0).contiguous(),
                'o': ly.self_attn.o_proj.weight.data.contiguous(),
                'down': ly.mlp.down_proj.weight.data.contiguous(),
                'in_norm': ly.input_layernorm.weight.data.contiguous(),
                'post_norm': ly.post_attention_layernorm.weight.data.contiguous(),
                'qn': ly.self_attn.q_norm.weight.data.contiguous(),
                'kn': ly.self_attn.k_norm.weight.data.contiguous(),
            })
        self.final_norm = qm.norm.weight.data.contiguous()

        inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD))
        pos = torch.arange(MAX_S, device=dev, dtype=torch.float32)
        f = torch.outer(pos, inv)
        self.rc = f.cos().to(BF16); self.rs = f.sin().to(BF16)
        self._rope_tables = {}; self._buf = None
        self._attn_o_buf = None; self._attn_lse = None
        self._fp4_ready = False; self.WL_fp4 = None; self._fp4_act = None; self._alphas = None
        self._graph = None; self._graph_pool = None; self._graph_stream = None
        self._graph_bs = 0; self._graph_input = None; self._graph_output = None
        self._requested_active_rows = None
        self._active_rows = None
        self._packed_cu_seqlens = {}
        self._graph_post_spec = None
        self._graph_post_refs = None
        self._graph_post_selection = None
        self._graph_maskgit_spec = None
        self._graph_base_input = None
        self._graph_refresh_base = False
        self._graph_tokens = None
        self._graph_schedule = None
        self._graph_step = None
        self._graph_cache = OrderedDict()
        self._graph_cache_limit = 2
        self._graph_maskgit_states = OrderedDict()
        self._num_sms = torch.cuda.get_device_properties(dev).multi_processor_count

    def _get_rope_bs(self, B, S):
        key = (B, S)
        if key not in self._rope_tables:
            cos_s = torch.cat([self.rc[:S], self.rc[:S]], -1)
            sin_s = torch.cat([self.rs[:S], self.rs[:S]], -1)
            self._rope_tables[key] = (cos_s.repeat(B, 1).contiguous(), sin_s.repeat(B, 1).contiguous())
        return self._rope_tables[key]

    def _alloc(self, B, S):
        BS = B * S; d = self.device
        self._buf = {
            'h': torch.empty(BS, D, device=d, dtype=BF16),
            'h2': torch.empty(BS, D, device=d, dtype=BF16),
            'xn': torch.empty(BS, D, device=d, dtype=BF16),
            'tmp': torch.empty(BS, D, device=d, dtype=BF16),
            'Dq': torch.empty(BS, QKVD, device=d, dtype=BF16),
            'Dg': torch.empty(BS, GUD, device=d, dtype=BF16),
            'act': torch.empty(BS, FFN, device=d, dtype=BF16),
            'q_flat': torch.empty(BS * NH, HD, device=d, dtype=BF16),
            'k_flat': torch.empty(BS * NKV, HD, device=d, dtype=BF16),
            'q_temp': torch.empty(BS * NH, HD, device=d, dtype=BF16),
            'k_temp': torch.empty(BS * NKV, HD, device=d, dtype=BF16),
            'k_expanded': torch.empty(
                B, NH, S, HD, device=d, dtype=BF16),
            'v_expanded': torch.empty(
                B, NH, S, HD, device=d, dtype=BF16),
        }
        self._attn_o_buf = torch.empty(B, S, NH, HD, device=d, dtype=BF16)
        sq = ((S + 127) // 128) * 128
        self._attn_lse = torch.empty(B, NH, sq, device=d, dtype=torch.float32)

    def _alloc_fp4_act(self, BS):
        d = self.device
        self._fp4_act = {
            'inp_packed': torch.empty(BS, D // 2, dtype=torch.uint8, device=d),
            'inp_sf': torch.zeros(_swizzled_sf_bytes(BS, D), dtype=torch.uint8, device=d),
            'ao_packed': torch.empty(BS, NQK // 2, dtype=torch.uint8, device=d),
            'ao_sf': torch.zeros(_swizzled_sf_bytes(BS, NQK), dtype=torch.uint8, device=d),
            'act_packed': torch.empty(BS, FFN // 2, dtype=torch.uint8, device=d),
            'act_sf': torch.zeros(_swizzled_sf_bytes(BS, FFN), dtype=torch.uint8, device=d),
        }

    def _quantize_weight(self, w_bf16):
        N, K = w_bf16.shape; d = w_bf16.device
        packed = torch.empty(N, K // 2, dtype=torch.uint8, device=d)
        sf = torch.zeros(_swizzled_sf_bytes(N, K), dtype=torch.uint8, device=d)
        self.fvk.quantize_bf16_to_nvfp4_swizzled_mse(
            w_bf16.data_ptr(), packed.data_ptr(), sf.data_ptr(), N, K,
            torch.cuda.current_stream().cuda_stream)
        return packed, sf

    @torch.inference_mode()
    def calibrate(self, calibration_input):
        B, S = calibration_input.shape[:2]; BS = B * S
        self._alloc(B, S)
        self.WL_fp4 = []; self._alphas = []
        for li in range(L):
            wb = self.WL_bf16[li]; wf = {}; al = {}
            for key in ['in_norm', 'post_norm', 'qn', 'kn']:
                wf[key] = wb[key]
            for key, N, K in [('qkv', QKVD, D), ('gu', GUD, D), ('o', D, NQK), ('down', D, FFN)]:
                packed, sf = self._quantize_weight(wb[key])
                wf[key + '_packed'] = packed; wf[key + '_sf'] = sf; al[key] = 1.0
            self.WL_fp4.append(wf); self._alphas.append(al)
        self._fp4_ready = True
        log.info("FP4: weight quantization complete (%d layers)", L)

    @torch.inference_mode()
    def forward(self, inputs_embeds, attention_mask=None, **kw):
        if not self._fp4_ready: self.calibrate(inputs_embeds)
        fvk = self.fvk; st = torch.cuda.current_stream().cuda_stream
        B, S0, _ = inputs_embeds.shape; BS = B * S0
        if self._buf is None or self._buf['h'].shape[0] != BS: self._alloc(B, S0)
        if self._fp4_act is None or self._fp4_act['inp_packed'].shape[0] != BS: self._alloc_fp4_act(BS)
        b = self._buf; fa = self._fp4_act; cos_bs, sin_bs = self._get_rope_bs(B, S0)
        seqused_k = self._attention_seqused_k(attention_mask, B, S0)
        graph_input = inputs_embeds.to(BF16).contiguous().reshape(BS, D)
        if graph_input.data_ptr() != b['h'].data_ptr():
            b['h'].copy_(graph_input)
        fvk.rms_norm_to_nvfp4_swizzled_bf16(b['h'].data_ptr(), self.WL_fp4[0]['in_norm'].data_ptr(),
            fa['inp_packed'].data_ptr(), fa['inp_sf'].data_ptr(), BS, D, EPS, st)
        h_ptr = b['h'].data_ptr(); h2_ptr = b['h2'].data_ptr()
        for li in range(L):
            w = self.WL_fp4[li]; al = self._alphas[li]; is_last = (li == L - 1)
            next_in_norm_ptr = (self.WL_fp4[li + 1]['in_norm'].data_ptr() if not is_last else 0)
            _call_fp4_gemm(fvk, fa['inp_packed'].data_ptr(), w['qkv_packed'].data_ptr(),
                b['Dq'].data_ptr(), BS, QKVD, D, fa['inp_sf'].data_ptr(), w['qkv_sf'].data_ptr(),
                al['qkv'], _pick_gemm_variant(QKVD, D), st)
            self.fvo.omnivoice_qk_norm_rope_bf16(b['Dq'].data_ptr(), w['qn'].data_ptr(), w['kn'].data_ptr(),
                cos_bs.data_ptr(), sin_bs.data_ptr(),
                b['q_temp'].data_ptr(), b['k_temp'].data_ptr(), BS, NH, NKV, HD, QKVD, EPS, st)
            b['q_flat'], b['q_temp'] = b['q_temp'], b['q_flat']
            b['k_flat'], b['k_temp'] = b['k_temp'], b['k_flat']
            qr = b['q_flat'].view(B, S0, NH, HD).contiguous()
            kr = b['k_flat'].view(B, S0, NKV, HD).contiguous()
            vv = b['Dq'][:, NQK + KVD:].contiguous().reshape(B, S0, NKV, HD).contiguous()
            self._fa2_fwd(qr, kr, vv, B, S0, NH, NKV, HD, st, seqused_k)
            ao_flat = self._attn_o_buf.reshape(BS, NQK).contiguous()
            fvk.quantize_bf16_to_nvfp4_swizzled(ao_flat.data_ptr(), fa['ao_packed'].data_ptr(), fa['ao_sf'].data_ptr(), BS, NQK, st)
            _call_fp4_gemm(fvk, fa['ao_packed'].data_ptr(), w['o_packed'].data_ptr(), b['tmp'].data_ptr(),
                BS, D, NQK, fa['ao_sf'].data_ptr(), w['o_sf'].data_ptr(), al['o'], _pick_gemm_variant(D, NQK), st)
            fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16(h_ptr, b['tmp'].data_ptr(), h2_ptr,
                w['post_norm'].data_ptr(), fa['inp_packed'].data_ptr(), fa['inp_sf'].data_ptr(), BS, D, EPS, st)
            h_ptr, h2_ptr = h2_ptr, h_ptr
            _call_fp4_gemm(fvk, fa['inp_packed'].data_ptr(), w['gu_packed'].data_ptr(), b['Dg'].data_ptr(),
                BS, GUD, D, fa['inp_sf'].data_ptr(), w['gu_sf'].data_ptr(), al['gu'], _pick_gemm_variant(GUD, D), st)
            fvk.silu_mul_merged_to_nvfp4_swizzled_bf16(b['Dg'].data_ptr(), fa['act_packed'].data_ptr(), fa['act_sf'].data_ptr(), BS, FFN, st)
            _call_fp4_gemm(fvk, fa['act_packed'].data_ptr(), w['down_packed'].data_ptr(), b['tmp'].data_ptr(),
                BS, D, FFN, fa['act_sf'].data_ptr(), w['down_sf'].data_ptr(), al['down'], _pick_gemm_variant(D, FFN), st)
            if not is_last:
                fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16(h_ptr, b['tmp'].data_ptr(), h2_ptr,
                    next_in_norm_ptr, fa['inp_packed'].data_ptr(), fa['inp_sf'].data_ptr(), BS, D, EPS, st)
                h_ptr, h2_ptr = h2_ptr, h_ptr
            else:
                fvk.residual_add_rms_norm(h_ptr, b['tmp'].data_ptr(), self.final_norm.data_ptr(),
                    b['xn'].data_ptr(), BS, D, EPS, st)
        return (b['xn'].reshape(B, S0, D).to(inputs_embeds.dtype),)

    def _attention_seqused_k(self, attention_mask, B, S):
        if attention_mask is None:
            return None
        # OmniVoice's generation mask is dense over the valid prefix.  Padded
        # query rows are diagonal-only, but their outputs are never consumed;
        # for every consumed row this is exactly a per-batch K length.
        if attention_mask.ndim != 4 or attention_mask.shape[0] != B:
            raise ValueError(
                "OmniVoice FlashRT expects attention_mask shaped [B,1,S,S]")
        return attention_mask[:,0,0,:S].ne(0).sum(
            dim=-1,dtype=torch.int32).contiguous()

    def _fa2_fwd(self, q, k, v, B, S, NH, NKV, HD, stream, seqused_k=None):
        o = self._attn_o_buf
        args=(q.data_ptr(),k.data_ptr(),v.data_ptr(),o.data_ptr(),
              self._attn_lse.data_ptr())
        shape_args=(B,S,S,NH,NKV,HD,
            (q.stride(0),q.stride(1),q.stride(2)),
            (k.stride(0),k.stride(1),k.stride(2)),
            (v.stride(0),v.stride(1),v.stride(2)),
            (o.stride(0),o.stride(1),o.stride(2)),
            HD ** -0.5,self._num_sms,stream)
        if seqused_k is not None and hasattr(self.fa2,'fwd_bf16_seqused'):
            self.fa2.fwd_bf16_seqused(
                *args,seqused_k.data_ptr(),*shape_args)
        else:
            self.fa2.fwd_bf16(*args,0,0,*shape_args)

    def _sdpa_fwd(self, q, k, v, attention_mask):
        """PyTorch SDPA path used by the exact BF16 engine.

        This intentionally mirrors transformers' Qwen3 SDPA adapter,
        including explicit GQA expansion when a custom mask is present.
        """
        qh=q.transpose(1,2)
        kh=k.transpose(1,2)
        vh=v.transpose(1,2)
        groups=qh.shape[1]//kh.shape[1]
        if groups != 1:
            B,NKV,S,HD=kh.shape
            kh=kh[:,:,None,:,:].expand(
                B,NKV,groups,S,HD).reshape(B,NKV*groups,S,HD)
            vh=vh[:,:,None,:,:].expand(
                B,NKV,groups,S,HD).reshape(B,NKV*groups,S,HD)
        out=F.scaled_dot_product_attention(
            qh,kh,vh,attn_mask=attention_mask,
            dropout_p=0.0,scale=HD ** -0.5,is_causal=False)
        self._attn_o_buf.copy_(out.transpose(1,2))

    def _sdpa_fwd_expanded(self, q, k, v, attention_mask):
        qh=q.transpose(1,2)
        out=F.scaled_dot_product_attention(
            qh,k,v,attn_mask=attention_mask,
            dropout_p=0.0,scale=HD ** -0.5,is_causal=False)
        self._attn_o_buf.copy_(out.transpose(1,2))

    def _sdpa_fwd_packed(self, q, k, v, batch, seq, active_rows):
        key=(batch,seq,active_rows)
        cu_seqlens=self._packed_cu_seqlens.get(key)
        if cu_seqlens is None:
            # Exact B=1 CFG packing: conditional S rows followed by the
            # valid unconditional prefix.
            cu_seqlens=torch.tensor(
                [0,seq,active_rows],device=self.device,dtype=torch.int32)
            self._packed_cu_seqlens[key]=cu_seqlens
        second_seq=active_rows-seq
        compact_bucket=(
            (seq+31)//32==3 and
            (second_seq+31)//32==2)
        if (batch==2 and compact_bucket and
                hasattr(
                    self.fvo,
                    "omnivoice_exact_packed_attention_compact_bf16")):
            out=self._attn_o_buf.view(-1,NH,HD)[:active_rows]
            self.fvo.omnivoice_exact_packed_attention_compact_bf16(
                q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),
                active_rows,seq,NH,HD,HD ** -0.5,
                torch.cuda.current_stream().cuda_stream)
            return out.view(1,active_rows,NH,HD)
        if hasattr(self.fvo,"omnivoice_exact_packed_attention_bf16"):
            out=self._attn_o_buf.view(-1,NH,HD)[:active_rows]
            self.fvo.omnivoice_exact_packed_attention_bf16(
                q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),
                cu_seqlens.data_ptr(),active_rows,seq,batch,NH,HD,
                HD ** -0.5,torch.cuda.current_stream().cuda_stream)
            return out.view(1,active_rows,NH,HD)
        return torch.ops.aten._efficient_attention_forward(
            q,k,v,None,cu_seqlens,cu_seqlens,
            seq,seq,0.0,0,False,scale=HD ** -0.5)[0]

    def _graph_post_key(self):
        spec=self._graph_post_spec
        if spec is None:
            return None
        head,tl,gs,mask_id,layer_penalty=spec
        return (
            head.weight.data_ptr(),
            0 if head.bias is None else head.bias.data_ptr(),
            tl,float(gs),int(mask_id),
            layer_penalty.data_ptr())

    def _graph_maskgit_key(self):
        spec=self._graph_maskgit_spec
        if spec is None:
            return None
        return (
            spec["tokens"].data_ptr(),
            spec["embedding"].data_ptr(),
            spec["offsets"].data_ptr(),
            spec["schedule"].data_ptr(),
            spec["step"].data_ptr(),
            int(spec["num_codebooks"]),
            int(spec["token_stride"]),
            int(spec["target_length"]),
            int(spec["schedule"].numel()),
            float(spec["position_temperature"]),
        )

    def _invalidate_graph(self):
        """Release captured-only refs before a shape/spec recapture."""
        graph=self._graph
        if graph is None:
            return
        # Recapture is a cold path. Waiting here prevents reset from racing a
        # prior request, while dropping graph-pool outputs avoids PyTorch's
        # caching-allocator use-count assertion on the next capture.
        torch.cuda.current_stream(device=self.device).synchronize()
        self._graph_output=None
        self._graph_post_refs=None
        self._graph_post_selection=None
        for key,state in tuple(self._graph_cache.items()):
            if state["_graph"] is graph:
                state["_graph_output"]=None
                state["_graph_post_refs"]=None
                state["_graph_post_selection"]=None
                del self._graph_cache[key]
                break
        graph.reset()
        self._graph=None

    _GRAPH_STATE_FIELDS=(
        "_buf","_attn_o_buf","_attn_lse",
        "_graph","_graph_pool","_graph_stream",
        "_graph_input","_graph_base_input","_graph_attention_mask",
        "_graph_output","_graph_post_refs","_graph_post_selection",
        "_graph_shape","_active_rows",
    )

    def _save_graph_state(self):
        shape=self._graph_shape
        self._graph_cache[shape]={
            field:getattr(self,field,None)
            for field in self._GRAPH_STATE_FIELDS
        }
        self._graph_cache.move_to_end(shape)
        while len(self._graph_cache)>self._graph_cache_limit:
            _,state=self._graph_cache.popitem(last=False)
            old_graph=state["_graph"]
            if old_graph is self._graph:
                continue
            torch.cuda.current_stream(device=self.device).synchronize()
            state["_graph_output"]=None
            state["_graph_post_refs"]=None
            state["_graph_post_selection"]=None
            old_graph.reset()

    def _activate_graph_state(self, shape):
        state=self._graph_cache.get(shape)
        if state is None:
            return False
        for field,value in state.items():
            setattr(self,field,value)
        self._graph_cache.move_to_end(shape)
        return True

    def _detach_graph_state(self):
        # Cached tensors and the graph keep the old shape alive. Clear only
        # the active aliases so _capture_graph allocates an independent pool.
        for field in self._GRAPH_STATE_FIELDS:
            setattr(self,field,None)

    def _run_graph_pre(self):
        spec=self._graph_maskgit_spec
        if spec is None:
            return
        # The residual workspace aliases graph_input and is overwritten by
        # every forward. One captured kernel restores the request's constant
        # text/prompt rows and regenerates both changing target spans from
        # graph-owned tokens.
        tl=int(spec["target_length"])
        seq=self._graph_input.shape[1]
        self.fvo.omnivoice_prepare_cfg_embeddings_bf16(
            spec["tokens"].data_ptr(),
            spec["embedding"].data_ptr(),
            spec["offsets"].data_ptr(),
            self._graph_base_input.data_ptr(),
            self._graph_input.data_ptr(),
            int(spec["num_codebooks"]),int(spec["token_stride"]),
            seq,tl,D,torch.cuda.current_stream().cuda_stream)

    def _run_graph_post(self, hidden):
        spec=self._graph_post_spec
        if spec is None:
            return None
        head,tl,gs,mask_id,layer_penalty=spec
        packed_hidden=hidden.reshape(-1,D)[
            hidden.shape[1]-tl:hidden.shape[1]+tl]
        packed_logits=head(packed_hidden)
        num_codebooks=layer_penalty.shape[1]
        vocab_size=packed_logits.shape[-1]//num_codebooks
        pred=torch.empty(
            (1,num_codebooks,tl),
            device=packed_logits.device,dtype=torch.int64)
        scores=torch.empty_like(pred,dtype=torch.float32)
        # One warp-per-row kernel mirrors all three PyTorch FP32
        # PersistentSoftmax reductions, preserves CFG rounding points, and
        # selects the non-mask token without materializing permuted logits or
        # any log-probability tensors.
        self.fvo.omnivoice_logits_bf16_cfg_argmax_scores_f32(
            packed_logits.data_ptr(),layer_penalty.data_ptr(),
            pred.data_ptr(),scores.data_ptr(),
            scores.numel(),vocab_size,tl,mask_id,gs,
            torch.cuda.current_stream().cuda_stream)
        uniforms=None
        maskgit=self._graph_maskgit_spec
        if maskgit is not None:
            temperature=float(maskgit["position_temperature"])
            common=(
                pred.data_ptr(),maskgit["tokens"].data_ptr(),
                maskgit["schedule"].data_ptr(),
                maskgit["step"].data_ptr(),scores.numel(),
                maskgit["schedule"].numel(),mask_id)
            if temperature>0.0:
                # CUDA Graph RNG advances the default generator on every
                # replay. _capture_graph restores the pre-capture RNG state,
                # so the first replay consumes exactly the uniforms that the
                # original eager torch.rand_like(scores) would have produced.
                uniforms=torch.rand_like(scores)
                self.fvo.omnivoice_maskgit_topk_gumbel_scheduled_update(
                    scores.data_ptr(),uniforms.data_ptr(),*common,
                    temperature,torch.cuda.current_stream().cuda_stream)
            else:
                self.fvo.omnivoice_maskgit_topk_scheduled_update(
                    scores.data_ptr(),*common,
                    torch.cuda.current_stream().cuda_stream)
        # Only graph outputs observed by Python need owning references.  The
        # graph-private allocator keeps the head/log-softmax intermediates at
        # their captured addresses, and releasing their Tensor wrappers lets
        # their storage be reused by later captures in the same private pool.
        return None,None,None,pred,scores,uniforms

    @torch.inference_mode()
    def _capture_graph(self, inputs_embeds, attention_mask=None):
        if self._graph is not None: return
        B, S, _ = inputs_embeds.shape
        BS = B * S
        requested = self._requested_active_rows
        self._active_rows = (
            requested if requested is not None and 0<requested<=BS
            else BS)
        if self._buf is None or self._buf['h'].shape[0] != BS:
            self._alloc(B, S)
        self._graph_stream = torch.cuda.Stream(device=self.device)
        if self._graph_pool is None: self._graph_pool = torch.cuda.graph_pool_handle()
        # Alias the graph input to the forward workspace.  The caller copies
        # embeddings into this buffer before replay, so forward() can consume
        # them directly instead of doing a second device-to-device copy.
        self._graph_input = self._buf['h'].view(B, S, D)
        self._graph_base_input=(
            inputs_embeds.to(BF16).contiguous().clone()
            if self._graph_maskgit_spec is not None else None)
        self._graph_attention_mask = (
            None if B==2 and S<self._active_rows<BS else
            attention_mask.clone() if attention_mask is not None else None)
        maskgit=self._graph_maskgit_spec
        token_snapshot=(
            maskgit["tokens"].clone() if maskgit is not None else None)
        step_snapshot=(
            maskgit["step"].clone() if maskgit is not None else None)
        rng_state=(
            torch.cuda.get_rng_state(self.device)
            if maskgit is not None and
            float(maskgit["position_temperature"])>0.0 else None)
        gs = self._graph_stream; gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs):
            for _ in range(3):
                if maskgit is not None:
                    maskgit["tokens"].copy_(token_snapshot)
                    maskgit["step"].copy_(step_snapshot)
                    self._run_graph_pre()
                else:
                    self._graph_input.copy_(
                        inputs_embeds.to(BF16).contiguous())
                warm=self.forward(
                    self._graph_input,self._graph_attention_mask)
                self._run_graph_post(warm[0])
            if maskgit is not None:
                maskgit["tokens"].copy_(token_snapshot)
                maskgit["step"].copy_(step_snapshot)
                self._run_graph_pre()
            else:
                self._graph_input.copy_(inputs_embeds.to(BF16).contiguous())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._graph_pool, stream=gs):
                self._run_graph_pre()
                out = self.forward(
                    self._graph_input,self._graph_attention_mask)
                post_refs=self._run_graph_post(out[0])
            # forward() already writes into a persistent workspace.  Returning
            # its view avoids another copy at the end of every graph replay.
            self._graph_output = out[0]
            self._graph_post_refs=post_refs
            self._graph_post_selection=(
                None if post_refs is None else
                (post_refs[3],post_refs[4]))
        torch.cuda.current_stream().wait_stream(gs)
        if maskgit is not None:
            maskgit["tokens"].copy_(token_snapshot)
            maskgit["step"].copy_(step_snapshot)
        if rng_state is not None:
            # Warm-up and capture must be invisible to seeded generation.
            torch.cuda.set_rng_state(rng_state,self.device)
        self._graph = g
        self._graph_shape = (
            B,S,attention_mask is not None,self._active_rows,
            self._graph_post_key(),self._graph_maskgit_key())
        self._save_graph_state()
        self._graph_refresh_base=False
        log.info("CUDA Graph captured (B=%d,S=%d)", B, S)

    @torch.inference_mode()
    def forward_graph(self, inputs_embeds, attention_mask=None, **kw):
        requested=self._requested_active_rows
        active=(requested if requested is not None
                else inputs_embeds.shape[0]*inputs_embeds.shape[1])
        shape=(
            *inputs_embeds.shape[:2],attention_mask is not None,active,
            self._graph_post_key(),self._graph_maskgit_key())
        if self._graph is None or shape != getattr(self,'_graph_shape',None):
            if not self._activate_graph_state(shape):
                self._detach_graph_state()
                self._capture_graph(inputs_embeds,attention_mask)
        graph_input=inputs_embeds.to(BF16).contiguous()
        B,S=inputs_embeds.shape[:2]
        if self._graph_maskgit_spec is not None:
            if self._graph_refresh_base:
                self._graph_base_input.copy_(graph_input)
                self._graph_refresh_base=False
        elif B==2 and S<active<B*S:
            # The graph aliases its input with the residual workspace, so all
            # valid rows must be restored before replay.  The padded
            # unconditional tail is never read and can still be skipped.
            self._graph_input.view(-1,D)[:active].copy_(
                graph_input.view(-1,D)[:active])
        else:
            self._graph_input.copy_(graph_input)
        if attention_mask is not None and self._graph_attention_mask is not None:
            self._graph_attention_mask.copy_(attention_mask)
        self._graph.replay()
        return (self._graph_output.to(inputs_embeds.dtype),)


# ═══════════════════════════════════════════════════════════════
# FlashRTLlmBF16 — BF16 forward engine (no quantization)
# ═══════════════════════════════════════════════════════════════

class FlashRTLlmBF16(FlashRTLlm):
    """Exact BF16 FlashRT — fused QK/RoPE, PyTorch SDPA, CUDA Graph."""

    def _use_qkv_heuristic(self, rows):
        # cuBLASLt heuristic rank 4 keeps rank 0's BF16 reduction order but
        # is faster for the 112..128-row short-CFG bucket on SM120.
        return (
            112<=rows<=128 and
            hasattr(self.fvo,"omnivoice_bf16_matmul_heuristic_algo"))

    def _use_dual_swiglu(self, rows):
        # The dual-GEMM tile wins for the short CFG regime and again after
        # four 64-row tiles. cuBLASLt is faster around 208..256 rows.
        fvo=getattr(self,"fvo",None)
        return (
            fvo is not None and
            hasattr(fvo,"omnivoice_bf16_dual_swiglu") and
            ((104<=rows<=192) or rows>=272))

    def _alloc(self, B, S):
        """Allocate only the exact-BF16 workspaces and alias disjoint phases."""
        BS=B*S
        work_rows=(
            self._active_rows
            if self._active_rows is not None and
            0<self._active_rows<=BS else BS)
        d=self.device
        dual_swiglu=self._use_dual_swiglu(work_rows)
        projection_width=QKVD if dual_swiglu else GUD
        projection=torch.empty(
            work_rows*projection_width,device=d,dtype=BF16)
        self._buf={
            "h":torch.empty(BS,D,device=d,dtype=BF16),
            "xn":torch.empty(BS,D,device=d,dtype=BF16),
            "tmp":torch.empty(work_rows,D,device=d,dtype=BF16),
            # QKV, attention output, and Gate/Up are never live together.
            # Give each a contiguous view of the same largest projection
            # workspace instead of retaining three shape-specific buffers.
            "projection":projection,
            "Dq":projection[:work_rows*QKVD].view(work_rows,QKVD),
            "Dg":(
                None if dual_swiglu else
                projection[:work_rows*GUD].view(work_rows,GUD)),
            "act":torch.empty(work_rows,FFN,device=d,dtype=BF16),
            # Q is overwritten by the next layer only after attention has
            # consumed it, so the BF16 path needs no ping-pong Q buffer.
            "q_flat":torch.empty(
                work_rows*NH,HD,device=d,dtype=BF16),
            "k_expanded":torch.empty(
                1,work_rows,NH,HD,device=d,dtype=BF16),
            "v_expanded":torch.empty(
                1,work_rows,NH,HD,device=d,dtype=BF16),
        }
        attn_shape=(
            (B,S,NH,HD) if work_rows==BS else
            (1,work_rows,NH,HD))
        self._attn_o_buf=projection[
            :work_rows*NQK].view(*attn_shape)
        self._attn_lse=None

    @torch.inference_mode()
    def calibrate(self, calibration_input):
        B, S = calibration_input.shape[:2]
        self._alloc(B, S)
        # Down projection is consumed as [K,N]. Materialize that static
        # layout once: cuBLASLt keeps the same BF16 reduction order for this
        # shape while avoiding the strided-transpose operand path.
        for weights in self.WL_bf16:
            if "down_t" not in weights:
                weights["down_t"]=weights.pop(
                    "down").t().contiguous()
        self.WL_fp4 = self.WL_bf16
        self._alphas = [{'qkv':1.0,'o':1.0,'gu':1.0,'down':1.0} for _ in range(L)]
        self._fp4_ready = True; self._fp4_act = None
        log.info("BF16 engine: ready (%d layers, no quantization)", L)

    def _exact_rms_norm_into(self, x, weight, out, stream):
        self.fvo.omnivoice_rms_norm_exact_1024_bf16(
            x.data_ptr(),weight.data_ptr(),out.data_ptr(),
            x.shape[0],EPS,stream)

    def _residual_add_exact_rms_norm_into(
            self, residual, update, weight, out, stream):
        self.fvo.omnivoice_residual_add_rms_norm_exact_1024_bf16(
            residual.data_ptr(),update.data_ptr(),
            weight.data_ptr(),out.data_ptr(),residual.shape[0],
            EPS,stream)

    @torch.inference_mode()
    def forward(self, inputs_embeds, attention_mask=None, **kw):
        if not self._fp4_ready: self.calibrate(inputs_embeds)
        st = torch.cuda.current_stream().cuda_stream
        B, S0, _ = inputs_embeds.shape; BS = B * S0
        active_rows=(
            self._active_rows if self._active_rows is not None else BS)
        if self._buf is None or self._buf['h'].shape[0] != BS: self._alloc(B, S0)
        b = self._buf; cos_bs, sin_bs = self._get_rope_bs(B, S0)
        graph_input = inputs_embeds.to(BF16).contiguous().reshape(BS, D)
        if graph_input.data_ptr() != b['h'].data_ptr():
            b['h'].copy_(graph_input)
        self._exact_rms_norm_into(
            b['h'][:active_rows],self.WL_bf16[0]['in_norm'],
            b['xn'][:active_rows],st)
        for li in range(L):
            wb = self.WL_bf16[li]
            if self._use_qkv_heuristic(active_rows):
                self.fvo.omnivoice_bf16_matmul_heuristic_algo(
                    b['xn'].data_ptr(),wb['qkv'].data_ptr(),
                    b['Dq'].data_ptr(),active_rows,QKVD,D,
                    active_rows,4,st)
            else:
                torch.matmul(
                    b['xn'][:active_rows],wb['qkv'].t(),
                    out=b['Dq'][:active_rows])
            use_packed_attention=(
                B==2 and active_rows<BS and active_rows>S0)
            qk_kernel=(
                self.fvo.omnivoice_qk_norm_rope_expand_packed_kv_bf16
                if use_packed_attention else
                self.fvo.omnivoice_qk_norm_rope_expand_kv_bf16)
            qk_kernel(
                b['Dq'].data_ptr(),wb['qn'].data_ptr(),
                wb['kn'].data_ptr(),cos_bs.data_ptr(),sin_bs.data_ptr(),
                b['q_flat'].data_ptr(),b['k_expanded'].data_ptr(),
                b['v_expanded'].data_ptr(),B,S0,NH,NKV,HD,QKVD,
                active_rows,EPS,st)
            if use_packed_attention:
                qr=b['q_flat'][:active_rows*NH].view(
                    1,active_rows,NH,HD)
                kr=b['k_expanded'].view(-1,NH,HD)[:active_rows].view(
                    1,active_rows,NH,HD)
                vr=b['v_expanded'].view(-1,NH,HD)[:active_rows].view(
                    1,active_rows,NH,HD)
                ao_flat=self._sdpa_fwd_packed(
                    qr,kr,vr,B,S0,active_rows).reshape(active_rows,NQK)
            else:
                qr=b['q_flat'].view(B,S0,NH,HD).contiguous()
                self._sdpa_fwd_expanded(
                    qr,b['k_expanded'].view(B,NH,S0,HD),
                    b['v_expanded'].view(B,NH,S0,HD),attention_mask)
                ao_flat=self._attn_o_buf.reshape(BS,NQK).contiguous()
            if active_rows < BS:
                # PyTorch changes its BF16 reduction algorithm when this
                # projection is shortened directly, which is enough to alter
                # MaskGIT's later token choices.  Reuse the full padded
                # shape's cuBLASLt algorithm while computing only valid rows:
                # same reduction order, without the padded-tail work.
                self.fvo.omnivoice_bf16_matmul_reference_algo(
                    ao_flat.data_ptr(),wb['o'].data_ptr(),
                    b['tmp'].data_ptr(),
                    active_rows,D,NQK,BS,st)
            else:
                torch.matmul(ao_flat,wb['o'].t(),out=b['tmp'])
            self._residual_add_exact_rms_norm_into(
                b['h'][:active_rows],b['tmp'][:active_rows],
                wb['post_norm'],b['xn'][:active_rows],st)
            if self._use_dual_swiglu(active_rows):
                status=self.fvo.omnivoice_bf16_dual_swiglu(
                    b['xn'].data_ptr(),wb['gu'].data_ptr(),
                    wb['gu'][FFN:].data_ptr(),b['act'].data_ptr(),
                    active_rows,FFN,D,st)
                if status!=0:
                    raise RuntimeError(
                        f"OmniVoice dual SwiGLU failed: {status}")
            else:
                torch.matmul(
                    b['xn'][:active_rows],wb['gu'].t(),
                    out=b['Dg'][:active_rows])
                self.fvo.omnivoice_silu_mul_merged_bf16(
                    b['Dg'].data_ptr(),b['act'].data_ptr(),
                    active_rows,FFN,st)
            torch.matmul(
                b['act'][:active_rows],wb['down_t'],
                out=b['tmp'][:active_rows])
            next_norm=(self.WL_bf16[li+1]['in_norm']
                       if li+1<L else self.final_norm)
            self._residual_add_exact_rms_norm_into(
                b['h'][:active_rows],b['tmp'][:active_rows],
                next_norm,b['xn'][:active_rows],st)
        return (b['xn'].reshape(B, S0, D).to(inputs_embeds.dtype),)


# ═══════════════════════════════════════════════════════════════
# MaskGIT loop — BF16 CFG + FP4 noCFG hybrid
# ═══════════════════════════════════════════════════════════════

def _make_cfg(gs, mask_id, apply_mask=True):
    def _cfg(c, u):
        # Match OmniVoice._predict_tokens_with_scoring exactly.  In
        # particular, the logits must be promoted to fp32 before either
        # log-softmax and the mask token is removed only after CFG.  The old
        # fused BF16 path masked ``c`` before CFG and rounded the combined
        # probabilities back to BF16; both differences can change MaskGIT's
        # confidence ordering even when the predicted token argmax agrees.
        if gs <= 0:
            cl=torch.nn.functional.log_softmax(c.float(),dim=-1)
            if apply_mask:
                cl[...,mask_id]=-float('inf')
            return cl
        cl=torch.nn.functional.log_softmax(c.float(),dim=-1)
        ul=torch.nn.functional.log_softmax(u.float(),dim=-1)
        if (cl.is_cuda and _fvo is not None and
                hasattr(_fvo,"omnivoice_cfg_combine_f32")):
            combined=torch.empty_like(cl)
            _fvo.omnivoice_cfg_combine_f32(
                cl.data_ptr(),ul.data_ptr(),combined.data_ptr(),
                cl.numel(),gs,torch.cuda.current_stream().cuda_stream)
        else:
            combined=cl+gs*(cl-ul)
        out=torch.log_softmax(combined,dim=-1)
        if apply_mask:
            out[...,mask_id]=-float('inf')
        return out
    return _cfg


def _optimize_maskgit(model, frt_fp4, cfg_ratio=1.0, bookend=False):
    from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample, _filter_top_k

    num_cb = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id
    _dev = model.device
    _arange_cb = torch.arange(num_cb, device=_dev).view(1, -1, 1)
    if not hasattr(model,"_frt_layer_penalties"):
        model._frt_layer_penalties={}
    def _sample_step(logits, step, tokens, schedules, c_lens, t_lens,
                     lp_fn, ct, pt, layer_penalty, B_n, u_logits=None,
                     precomputed_lp=None, precomputed_selection=None):
        for i in range(B_n):
            k = schedules[i][step]
            if k <= 0: continue
            cl, tl = c_lens[i], t_lens[i]
            if precomputed_selection is not None:
                pred,scores=precomputed_selection
                lp=None
            elif precomputed_lp is not None:
                lp=precomputed_lp
            else:
                lp = lp_fn(
                    logits[i:i+1,:,cl-tl:cl,:],
                    u_logits[i:i+1,:,:tl,:]
                ) if u_logits is not None else lp_fn(
                    logits[i:i+1,:,cl-tl:cl,:])
            if (precomputed_selection is None and
                    ct<=0 and lp.dtype==torch.float32 and
                    hasattr(_fvo,"omnivoice_argmax_scores_f32")):
                pred=torch.empty(
                    lp.shape[:-1],device=lp.device,dtype=torch.int64)
                scores=torch.empty_like(pred,dtype=torch.float32)
                _fvo.omnivoice_argmax_scores_f32(
                    lp.data_ptr(),layer_penalty.data_ptr(),
                    pred.data_ptr(),scores.data_ptr(),
                    scores.numel(),lp.shape[-1],tl,mask_id,
                    torch.cuda.current_stream().cuda_stream)
            elif precomputed_selection is None:
                pred = (
                    _gumbel_sample(_filter_top_k(lp,0.1),ct).argmax(-1)
                    if ct>0 else lp.argmax(-1))
                scores = lp.max(-1)[0] - layer_penalty
            st_ = tokens[i:i+1,:,:tl]
            if (pt>0 and B_n==1 and
                    scores.numel()<=MASKGIT_CUDA_SELECT_MAX_ELEMENTS and
                    hasattr(
                        _fvo,"omnivoice_maskgit_topk_gumbel_update")):
                uniforms=torch.rand_like(scores)
                _fvo.omnivoice_maskgit_topk_gumbel_update(
                    scores.data_ptr(),uniforms.data_ptr(),
                    pred.data_ptr(),st_.data_ptr(),
                    scores.numel(),k,mask_id,pt,
                    torch.cuda.current_stream().cuda_stream)
                continue
            if pt > 0:
                scores = _gumbel_sample(scores, pt)
            if (B_n==1 and
                    scores.numel()<=MASKGIT_CUDA_SELECT_MAX_ELEMENTS and
                    hasattr(_fvo,"omnivoice_maskgit_topk_update")):
                _fvo.omnivoice_maskgit_topk_update(
                    scores.data_ptr(),pred.data_ptr(),st_.data_ptr(),
                    scores.numel(),k,mask_id,
                    torch.cuda.current_stream().cuda_stream)
                continue
            scores.masked_fill_(st_!=mask_id, -float("inf"))
            _, idx = torch.topk(scores.flatten(), k)
            ft = st_.flatten(); ft[idx] = pred.flatten()[idx]; st_.copy_(ft.view_as(st_))
            tokens[i:i+1,:,:tl] = st_

    def _gen(task, gen_config):
        B = task.batch_size
        inputs = [model._prepare_inference_inputs(
            task.texts[i], task.target_lens[i], task.ref_texts[i],
            task.ref_audio_tokens[i], task.langs[i], task.instructs[i], gen_config.denoise) for i in range(B)]
        c_lens = [inp["input_ids"].size(2) for inp in inputs]
        max_c = max(c_lens); max_t = max(task.target_lens)
        gs = gen_config.guidance_scale; NS = gen_config.num_step
        cfg_step = min(int(NS * cfg_ratio), NS - 1 if bookend else NS)
        ct, pt, lpf = gen_config.class_temperature, gen_config.position_temperature, gen_config.layer_penalty_factor
        penalty_key=float(lpf)
        layer_penalty=model._frt_layer_penalties.get(penalty_key)
        if layer_penalty is None:
            layer_penalty=_arange_cb*lpf
            model._frt_layer_penalties[penalty_key]=layer_penalty

        ts = _get_time_steps(0.0, 1.0, NS, gen_config.t_shift).tolist()
        schedules = []
        for tl in task.target_lens:
            tm=tl*num_cb; rem=tm; sc=[]
            for step in range(NS):
                n = rem if step==NS-1 else min(math.ceil(tm*(ts[step+1]-ts[step])), rem)
                sc.append(int(n)); rem -= int(n)
            schedules.append(sc)

        fast_argmax=(
            ct<=0 and _fvo is not None and
            hasattr(_fvo,"omnivoice_argmax_scores_f32"))
        full_step_graph=(
            B==1 and cfg_step==NS and fast_argmax and
            getattr(model,"_frt",None) is not None and
            num_cb*max_t<=MASKGIT_CUDA_SELECT_MAX_ELEMENTS and
            hasattr(
                _fvo,"omnivoice_maskgit_topk_scheduled_update") and
            hasattr(
                _fvo,
                "omnivoice_maskgit_topk_gumbel_scheduled_update"))
        if full_step_graph:
            frt=model._frt
            token_shape=(1,num_cb,max_t)
            state_key=(token_shape,NS)
            maskgit_state=frt._graph_maskgit_states.get(state_key)
            if maskgit_state is None:
                maskgit_state={
                    "tokens":torch.empty(
                        token_shape,dtype=torch.long,device=_dev),
                    "schedule":torch.empty(
                        NS,dtype=torch.int32,device=_dev),
                    "step":torch.zeros(
                        (),dtype=torch.int32,device=_dev),
                }
                frt._graph_maskgit_states[state_key]=maskgit_state
                while len(frt._graph_maskgit_states)>8:
                    frt._graph_maskgit_states.popitem(last=False)
            else:
                frt._graph_maskgit_states.move_to_end(state_key)
            frt._graph_tokens=maskgit_state["tokens"]
            frt._graph_schedule=maskgit_state["schedule"]
            frt._graph_step=maskgit_state["step"]
            tokens=frt._graph_tokens
            tokens.fill_(mask_id)
            schedule_values=torch.tensor(
                schedules[0],dtype=torch.int32,device=_dev)
            frt._graph_schedule.copy_(schedule_values)
            frt._graph_step.zero_()
        else:
            tokens=torch.full(
                (B,num_cb,max_t),mask_id,dtype=torch.long,device=_dev)
        _cfg = _make_cfg(gs, mask_id, apply_mask=not fast_argmax)
        _model = model.__call__

        def _nocfg_lp(c_lg):
            lp = torch.nn.functional.log_softmax(c_lg, dim=-1)
            if not fast_argmax:
                lp[...,mask_id] = -float('inf')
            return lp

        # Phase 1: BF16 + CFG (B=2)
        bm = 2
        bid2 = torch.full((bm*B, num_cb, max_c), mask_id, dtype=torch.long, device=_dev)
        bam2 = torch.zeros((bm*B, max_c), dtype=torch.bool, device=_dev)
        battn2 = torch.zeros((bm*B, 1, max_c, max_c), dtype=torch.bool, device=_dev)
        for i, inp in enumerate(inputs):
            cl = c_lens[i]; bid2[i,:,:cl]=inp["input_ids"]; bam2[i,:cl]=inp["audio_mask"]; battn2[i,:,:cl,:cl]=True
            ul = task.target_lens[i]
            bid2[B+i,:,:ul]=inp["input_ids"][...,-ul:]; bam2[B+i,:ul]=inp["audio_mask"][...,-ul:]
            battn2[B+i,:,:ul,:ul]=True
            if max_c>ul: pd=torch.arange(ul,max_c,device=_dev); battn2[B+i,:,pd,pd]=True
        cached_embeds_b1 = (
            model._prepare_embed_inputs(bid2,bam2)
            if B==1 else None)
        codebook_offsets = model.codebook_layer_offsets.view(1,-1,1)
        if B==1 and getattr(model,"_frt",None) is not None:
            model._frt._requested_active_rows=c_lens[0]+task.target_lens[0]
            model._frt._graph_post_spec=(
                (model.audio_heads,task.target_lens[0],gs,mask_id,
                 layer_penalty)
                if fast_argmax else None)
            model._frt._graph_maskgit_spec=(
                {
                    "tokens":tokens,
                    "embedding":model.audio_embeddings.weight,
                    "offsets":model.codebook_layer_offsets,
                    "schedule":model._frt._graph_schedule,
                    "step":model._frt._graph_step,
                    "num_codebooks":num_cb,
                    "token_stride":tokens.stride(1),
                    "target_length":task.target_lens[0],
                    "position_temperature":pt,
                }
                if full_step_graph else None)
            model._frt._graph_refresh_base=full_step_graph
        elif getattr(model,"_frt",None) is not None:
            model._frt._requested_active_rows=None
            model._frt._graph_post_spec=None
            model._frt._graph_maskgit_spec=None
            model._frt._graph_refresh_base=False

        def _cfg_forward():
            # Keep the same 2*B batch geometry as upstream.  FlashRTLlm maps
            # the full boolean mask through SDPA, so valid unconditional rows
            # cannot see the padded tail.
            if B==1:
                tl=task.target_lens[0]
                if full_step_graph:
                    # The graph restores cached_embeds_b1, gathers the current
                    # token embeddings, and writes both CFG target spans
                    # before the first model kernel.
                    pass
                elif hasattr(_fvo,"omnivoice_embed_codebooks_bf16"):
                    _fvo.omnivoice_embed_codebooks_bf16(
                        tokens.data_ptr(),
                        model.audio_embeddings.weight.data_ptr(),
                        model.codebook_layer_offsets.data_ptr(),
                        cached_embeds_b1[
                            0,c_lens[0]-tl:c_lens[0]].data_ptr(),
                        cached_embeds_b1[1,:tl].data_ptr(),
                        num_cb,tokens.stride(1),tl,D,
                        torch.cuda.current_stream().cuda_stream)
                else:
                    target_embeds=model.audio_embeddings(
                        tokens[:,:,:tl]+codebook_offsets).sum(dim=1)
                    cached_embeds_b1[
                        0,c_lens[0]-tl:c_lens[0]].copy_(target_embeds[0])
                    cached_embeds_b1[1,:tl].copy_(target_embeds[0])
                embeds=cached_embeds_b1
            else:
                embeds=model._prepare_embed_inputs(bid2,bam2)
            hidden=model.llm(
                inputs_embeds=embeds,attention_mask=battn2)[0]
            if B==1:
                # With one request the conditional target tail is immediately
                # followed in memory by the unconditional target prefix.
                # View that contiguous span directly instead of cat + two
                # logits copies on every MaskGIT step.
                cl=c_lens[0]
                tl=task.target_lens[0]
                if fast_argmax:
                    selection=model._frt._graph_post_selection
                    return None,None,None,selection
                packed_hidden=hidden.reshape(-1,D)[cl-tl:cl+tl]
                packed_logits=model.audio_heads(packed_hidden).float()
                logits=packed_logits.view(
                    2,tl,num_cb,-1).permute(0,2,1,3)
                return logits[:1],logits[1:],None,None
            selected=[]
            for i in range(B):
                cl,tl=c_lens[i],task.target_lens[i]
                selected.append(hidden[i,cl-tl:cl])
            for i in range(B):
                tl=task.target_lens[i]
                selected.append(hidden[B+i,:tl])
            lengths=[x.shape[0] for x in selected]
            packed_hidden=torch.cat(selected,dim=0)
            packed_logits=model.audio_heads(packed_hidden).float()
            c_logits=torch.empty(
                (B,num_cb,max_t,packed_logits.shape[-1]//num_cb),
                device=_dev,dtype=torch.float32)
            u_logits=torch.empty_like(c_logits)
            off=0
            for i,tl in enumerate(lengths[:B]):
                part=packed_logits[off:off+tl].view(
                    tl,num_cb,-1).permute(1,0,2)
                c_logits[i,:,:tl].copy_(part);off+=tl
            for i,tl in enumerate(lengths[B:]):
                part=packed_logits[off:off+tl].view(
                    tl,num_cb,-1).permute(1,0,2)
                u_logits[i,:,:tl].copy_(part);off+=tl
            return c_logits,u_logits,None,None

        for step in range(cfg_step):
            if full_step_graph and step>0:
                # Shape checks, dtype/view normalization, and module wrapper
                # dispatch are invariant after the first replay. Launch the
                # already-validated graph directly on the remaining steps.
                model._frt._graph.replay()
                continue
            (logits,u_logits,precomputed_lp,
             precomputed_selection)=_cfg_forward()
            if full_step_graph:
                # Sampling, scheduled Top-K, and the in-place token update are
                # the tail of the same replay. The device step selects k and
                # advances without a per-step host write.
                continue
            _sample_step(logits, step, tokens, schedules, task.target_lens,
                         task.target_lens, _cfg, ct, pt, layer_penalty, B,
                         u_logits=u_logits,
                         precomputed_lp=precomputed_lp,
                         precomputed_selection=precomputed_selection)
            if cfg_step<NS:
                for i in range(B):
                    cl,tl = c_lens[i], task.target_lens[i]
                    bid2[i:i+1,:,cl-tl:cl]=tokens[i:i+1,:,:tl]
                    bid2[B+i:B+i+1,:,:tl]=tokens[i:i+1,:,:tl]

        # Phase 2: FP4 + noCFG (B=1)
        orig_fwd = model.llm.forward
        def _fp4_fwd(*a, **kw):
            e = kw.get('inputs_embeds')
            if e is not None: del kw['inputs_embeds']
            elif len(a) >= 2 and isinstance(a[1], torch.Tensor): e = a[1]
            else: return orig_fwd(*a, **kw)
            if frt_fp4._graph is not None: return frt_fp4.forward_graph(e, **kw)
            return frt_fp4.forward(e, **kw)
        model.llm.forward = _fp4_fwd
        try:
            bid1 = bid2[:B].clone(); bam1 = bam2[:B].clone(); battn1 = battn2[:B].clone()
            fp4_end = NS - 1 if bookend else NS
            for step in range(cfg_step, fp4_end):
                logits = _model(input_ids=bid1, audio_mask=bam1, attention_mask=battn1).logits
                _sample_step(logits, step, tokens, schedules, c_lens,
                             task.target_lens, _nocfg_lp, ct, pt,
                             layer_penalty, B)
                for i in range(B):
                    cl,tl = c_lens[i], task.target_lens[i]
                    bid1[i:i+1,:,cl-tl:cl]=tokens[i:i+1,:,:tl]
        finally:
            model.llm.forward = orig_fwd

        # Phase 3: BF16 + CFG bookend
        if bookend and fp4_end < NS:
            for i in range(B):
                cl,tl = c_lens[i], task.target_lens[i]
                bid2[i:i+1,:,cl-tl:cl]=tokens[i:i+1,:,:tl]; bid2[B+i:B+i+1,:,:tl]=tokens[i:i+1,:,:tl]
            step = NS - 1
            (logits,u_logits,precomputed_lp,
             precomputed_selection)=_cfg_forward()
            _sample_step(logits, step, tokens, schedules, task.target_lens,
                         task.target_lens, _cfg, ct, pt, layer_penalty, B,
                         u_logits=u_logits,
                         precomputed_lp=precomputed_lp,
                         precomputed_selection=precomputed_selection)

        return [tokens[i,:,:task.target_lens[i]] for i in range(B)]

    model._generate_iterative = _gen


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

_injected = False; _frt_bf16 = None; _frt_fp4 = None; _orig = None; _orig_gen = None


def _optimize_audio_postprocess(model):
    if hasattr(model,"_frt_original_post_process_audio"):
        return
    original=model._post_process_audio
    model._frt_original_post_process_audio=original

    def _post_process_audio(
            generated_audio,postprocess_output,ref_rms):
        if postprocess_output:
            generated_audio=_remove_silence_numpy_exact(
                generated_audio,model.sampling_rate,
                mid_sil=500,lead_sil=100,trail_sil=100)
        volume_op=None
        if ref_rms is not None and ref_rms<0.1:
            volume_op=("ref",ref_rms)
        elif ref_rms is None:
            peak=np.abs(generated_audio).max()
            if peak>1e-6:
                volume_op=("peak",peak)
        return _fade_and_pad_numpy_exact(
            generated_audio,sample_rate=model.sampling_rate,
            volume_op=volume_op)

    model._post_process_audio=_post_process_audio


def _optimize_audio_decoder(model):
    """Capture fixed-shape tokenizer decode calls without changing math."""
    tokenizer=getattr(model,"audio_tokenizer",None)
    quantizer=getattr(tokenizer,"quantizer",None)
    if (tokenizer is None or quantizer is None or
            hasattr(tokenizer,"_frt_original_decode")):
        return

    tokenizer._frt_original_decode=tokenizer.decode
    tokenizer._frt_original_quantizer_decode=quantizer.decode

    # Upstream starts the reduction with a freshly allocated CUDA scalar:
    # ``torch.tensor(0.0, device=...)``.  Besides one redundant add, that
    # CPU-to-device scalar construction is illegal during CUDA Graph capture.
    # Assigning the first quantizer result directly is bit-identical because
    # FP32/BF16 x + 0 is exactly x.
    def _decode_codebooks(codes):
        if codes.shape[0]==0:
            return tokenizer._frt_original_quantizer_decode(codes)
        out=quantizer.quantizers[0].decode(codes[0]).to(codes.device)
        for index in range(1,codes.shape[0]):
            out=out+quantizer.quantizers[index].decode(
                codes[index]).to(codes.device)
        return out

    quantizer.decode=_decode_codebooks
    tokenizer._frt_decoder_graphs={}
    tokenizer._frt_decoder_graph_pool=None

    @torch.inference_mode()
    def _decode_graph(audio_codes,return_dict=None):
        original=tokenizer._frt_original_decode
        if (not audio_codes.is_cuda or
                torch.cuda.is_current_stream_capturing()):
            return original(audio_codes,return_dict=return_dict)
        resolved=(
            tokenizer.config.return_dict
            if return_dict is None else bool(return_dict))
        key=(
            tuple(audio_codes.shape),audio_codes.dtype,
            audio_codes.device.index,resolved)
        entry=tokenizer._frt_decoder_graphs.get(key)
        if entry is None:
            # Bound graph memory for variable-length serving. Shapes beyond
            # the small hot set remain on the exact eager decoder path.
            if len(tokenizer._frt_decoder_graphs)>=4:
                return original(audio_codes,return_dict=resolved)
            static_codes=torch.empty_like(audio_codes)
            capture_stream=torch.cuda.Stream(device=audio_codes.device)
            capture_stream.wait_stream(torch.cuda.current_stream())
            if tokenizer._frt_decoder_graph_pool is None:
                tokenizer._frt_decoder_graph_pool=(
                    torch.cuda.graph_pool_handle())
            with torch.cuda.stream(capture_stream):
                static_codes.copy_(audio_codes)
                for _ in range(3):
                    captured=original(
                        static_codes,return_dict=resolved)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                        graph,
                        pool=tokenizer._frt_decoder_graph_pool,
                        stream=capture_stream):
                    captured=original(
                        static_codes,return_dict=resolved)
            torch.cuda.current_stream().wait_stream(capture_stream)
            entry=(graph,static_codes,captured)
            tokenizer._frt_decoder_graphs[key]=entry
        graph,static_codes,captured=entry
        static_codes.copy_(audio_codes)
        graph.replay()
        return captured

    tokenizer.decode=_decode_graph

def _check_kernels():
    """Verify all required FlashRT kernel modules are available.

    Raises RuntimeError with rebuild instructions if any kernel is missing.
    Call before model loading to fail fast.
    """
    missing = []
    if _fvk is None:
        missing.append("flash_rt_kernels — FlashRT base not installed")
    else:
        for sym in (
            "fp4_w4a16_gemm_sm120_bf16out",
            "rms_norm",
            "residual_add_rms_norm_to_nvfp4_swizzled_bf16",
            "silu_mul_merged_to_nvfp4_swizzled_bf16",
            "quantize_bf16_to_nvfp4_swizzled_mse",
        ):
            if not hasattr(_fvk, sym):
                missing.append("flash_rt_kernels." + sym)
    if _fvo is None:
        missing.append(
            "flash_rt_omnivoice not found; "
            "rebuild FlashRT with: cmake -DFLASHRT_ENABLE_OMNIVOICE=ON -DGPU_ARCH=120")
    else:
        for sym in (
            "omnivoice_bf16_dual_swiglu",
            "omnivoice_bf16_matmul_reference_algo",
            "omnivoice_bf16_matmul_heuristic_algo",
            "omnivoice_exact_packed_attention_bf16",
            "omnivoice_exact_packed_attention_compact_bf16",
            "omnivoice_embed_codebooks_bf16",
            "omnivoice_prepare_cfg_embeddings_bf16",
            "omnivoice_argmax_scores_f32",
            "omnivoice_maskgit_topk_update",
            "omnivoice_maskgit_topk_gumbel_update",
            "omnivoice_maskgit_topk_scheduled_update",
            "omnivoice_maskgit_topk_gumbel_scheduled_update",
            "omnivoice_cfg_logsoftmax_bf16",
            "omnivoice_qk_norm_rope_bf16",
            "omnivoice_qk_norm_rope_expand_kv_bf16",
            "omnivoice_qk_norm_rope_expand_packed_kv_bf16",
            "omnivoice_rms_apply_exact_bf16",
            "omnivoice_rms_apply_from_mean_exact_bf16",
            "omnivoice_square_bf16_f32",
            "omnivoice_residual_add_square_bf16_f32",
            "omnivoice_square_mean_1024_bf16_f32",
            "omnivoice_residual_add_square_mean_1024_bf16_f32",
            "omnivoice_rms_norm_exact_1024_bf16",
            "omnivoice_residual_add_rms_norm_exact_1024_bf16",
            "omnivoice_silu_mul_merged_bf16",
            "omnivoice_expand_kv_bf16",
        ):
            if not hasattr(_fvo, sym):
                missing.append("flash_rt_omnivoice." + sym)
    if missing:
        raise RuntimeError(
            "OmniVoice FlashRT engine: required kernel symbols missing.\n"
            + "\n".join("  missing: " + s for s in missing)
        )

def inject(m, cfg_ratio=1.0, bookend=False):
    """Patch OmniVoice model for FlashRT acceleration.

    Args:
        m: OmniVoice model instance
        cfg_ratio: Fraction of steps using exact BF16 CFG. The default 1.0
            matches the PyTorch BF16 token path; lower values trade quality
            for speed by using FP4 without CFG for the remaining steps.
        bookend: If True, the final step also uses BF16 CFG
    """
    global _injected, _frt_bf16, _frt_fp4, _orig, _orig_gen
    if _injected: return
    _check_kernels()

    _orig = m.llm.forward
    _frt_bf16 = FlashRTLlmBF16(m.llm, str(m.device))
    needs_fp4 = cfg_ratio < 1.0
    _frt_fp4 = FlashRTLlm(m.llm, str(m.device)) if needs_fp4 else None

    c2 = torch.randn(2, 178, 1024, device=m.device, dtype=BF16) * 0.02
    _frt_bf16.calibrate(c2)
    if _frt_fp4 is not None:
        _frt_fp4.calibrate(c2)

    if _frt_fp4 is not None:
        _frt_fp4.WL_bf16 = None
    _frt_bf16.WL_fp4 = None; _frt_bf16._fp4_act = None; _frt_bf16._alphas = None
    for layer in m.llm.layers:
        for param in layer.parameters(recurse=True):
            param.data = torch.empty(0, device=param.device, dtype=param.dtype)
    torch.cuda.empty_cache()

    if _frt_fp4 is not None:
        c1 = torch.randn(1, 178, 1024, device=m.device, dtype=BF16) * 0.02
        for _ in range(3): _frt_fp4.forward(c1)
        torch.cuda.synchronize(); _frt_fp4._capture_graph(c1)
        for _ in range(3): _frt_fp4.forward_graph(c1); torch.cuda.synchronize()
    _frt_bf16._graph = None

    def _fwd_bf16(e, **kw): return _frt_bf16.forward_graph(e, **kw)
    def p(*a, **kw):
        e = kw.get('inputs_embeds')
        if e is not None: del kw['inputs_embeds']; return _fwd_bf16(e, **kw)
        if len(a) >= 2 and isinstance(a[1], torch.Tensor): return _fwd_bf16(a[1], **kw)
        return _orig(*a, **kw)
    m.llm.forward = p; m._frt = _frt_bf16

    _orig_gen = m._generate_iterative
    _optimize_maskgit(m, _frt_fp4, cfg_ratio, bookend)
    _optimize_audio_decoder(m)
    _optimize_audio_postprocess(m)
    _injected = True
    log.info("Engine injected: BF16 CFG(%.0f%%) + FP4 noCFG + bookend=%s", cfg_ratio*100, bookend)

def free_encoder(m):
    """Release AudioTokenizer encoder weights (~600 MB).

    After freeing, new voice clone prompts cannot be created.
    """
    if getattr(m, '_encoder_freed', False): return 0
    freed = 0
    for name in ['semantic_model', 'acoustic_encoder', 'encoder_semantic']:
        mod = getattr(m.audio_tokenizer, name, None)
        if mod is not None:
            s = sum(p.numel()*p.element_size() for p in mod.parameters())
            freed += s
            for p in mod.parameters(): p.data = torch.empty(0, device=p.device, dtype=p.dtype)
            log.info("Freed %s (%d MB)", name, s//1024//1024)
    torch.cuda.empty_cache()
    m._encoder_freed = True
    return freed // 1024 // 1024

def eject(m):
    """Restore original forward and generate methods."""
    global _injected, _orig, _orig_gen
    if not _injected: return
    if _orig is not None: m.llm.forward = _orig
    if _orig_gen is not None: m._generate_iterative = _orig_gen
    if hasattr(m,"_frt_original_post_process_audio"):
        m._post_process_audio=m._frt_original_post_process_audio
        delattr(m,"_frt_original_post_process_audio")
    tokenizer=getattr(m,"audio_tokenizer",None)
    if tokenizer is not None and hasattr(
            tokenizer,"_frt_original_decode"):
        tokenizer.decode=tokenizer._frt_original_decode
        tokenizer.quantizer.decode=(
            tokenizer._frt_original_quantizer_decode)
        for name in (
                "_frt_original_decode",
                "_frt_original_quantizer_decode",
                "_frt_decoder_graphs",
                "_frt_decoder_graph_pool"):
            if hasattr(tokenizer,name):
                delattr(tokenizer,name)
    _injected = False
