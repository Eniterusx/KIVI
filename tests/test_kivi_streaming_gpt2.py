import torch
import torch.nn as nn
from context.kivi_gpt2 import GPT2AttentionKIVI, KIVIQuantConfig

class SimpleConfig:
    def __init__(self, n_embd, n_head, n_ctx=1024, attn_pdrop=0.0, resid_pdrop=0.0):
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

def test_kivi_gpt2_streaming():
    print("Testing KIVI GPT-2 Streaming Logic...")
    
    # Setup config
    residual_length = 4
    n_embd = 32
    n_head = 4
    
    config = SimpleConfig(n_embd=n_embd, n_head=n_head)
    
    quant_config = KIVIQuantConfig(
        k_bits=2, 
        v_bits=2, 
        group_size=4,   # Must match residual_length for test
        residual_length=residual_length, 
        use_kivi=True
    )
    
    # Initialize Attention
    attn = GPT2AttentionKIVI(config, quant_config=quant_config)
    attn.eval()
    
    # Mock input
    bsz = 1
    seqlen = 10
    
    # Create a sequence of 10 tokens
    x = torch.randn(bsz, seqlen, n_embd)
    
    print("\n--- Phase 1: Streaming Process ---")
    past_kv = None
    
    # Process token by token
    for t in range(seqlen):
        token = x[:, t:t+1, :]
        print(f"\nTime step {t}:")
        
        output, past_kv = attn(token, past_kv=past_kv, use_cache=True)
        
        if past_kv is None:
             print("ERROR: past_kv returned None")
             break

        # Unpack cache
        (
            key_quant, key_residual, _, _, 
            value_quant, value_residual, _, _, 
            kv_seq_len
        ) = past_kv
        
        # Verify residual Growth
        k_res_len = key_residual.size(2) if key_residual is not None else 0
        v_res_len = value_residual.size(2) if value_residual is not None else 0
        
        print(f"  Key residual len: {k_res_len}")
        print(f"  Val residual len: {v_res_len}")
        print(f"  Quantized keys present: {key_quant is not None}")
        print(f"  Quantized vals present: {value_quant is not None}")
        
        # CHECK 1: Key Flush Logic (Batch flush when == R)
        if (t + 1) % residual_length == 0:
            if key_residual is not None:
                # Based on my imp, key_residual is set to empty tensor (size 0)
                if key_residual.size(2) != 0:
                     print(f"  WARNING: Key residual should be empty after flush at t={t}")
                else:
                     print(f"  SUCCESS: Key flushed at t={t}")
            else:
                print(f"  SUCCESS: Key flushed at t={t}")
        else:
            # Should be accumulating
            expected_len = (t + 1) % residual_length
            if k_res_len != expected_len:
                 print(f"  ERROR: Key residual len {k_res_len} != expected {expected_len}")

        # CHECK 2: Value Flush Logic (FIFO pop when > R)
        expected_v_len = min(t + 1, residual_length)
        if v_res_len != expected_v_len:
             print(f"  ERROR: Val residual len {v_res_len} != expected {expected_v_len}")

    print("\nTest Complete")

if __name__ == "__main__":
    with torch.no_grad():
        test_kivi_gpt2_streaming()
