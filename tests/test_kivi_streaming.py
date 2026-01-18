import torch
import torch.nn as nn
from context.kivi_smollm import SmolLM2Config, MHA, apply_rope, pre_compute_rope

def test_kivi_streaming_logic():
    print("Testing KIVI Streaming Logic...")
    
    # Setup config
    residual_length = 4
    config = SmolLM2Config(
        n_embd=32, n_head=4, n_kv_heads=2, 
        residual_length=residual_length, 
        use_kivi=True,
        k_bits=2, v_bits=2, group_size=4,
        dtype=torch.float32
    )
    
    # Initialize MHA
    mha = MHA(config)
    mha.eval()
    
    # Mock input
    bsz = 1
    seqlen = 10
    head_dim = config.n_embd // config.n_head
    
    # Create a sequence of 10 tokens
    x = torch.randn(bsz, seqlen, config.n_embd)
    
    print("\n--- Phase 1: Streaming Process ---")
    past_kv = None
    
    # Process token by token
    for t in range(seqlen):
        token = x[:, t:t+1, :]
        print(f"\nTime step {t}:")
        
        output, past_kv = mha(token, past_kv=past_kv, use_cache=True)
        
        # Unpack cache
        (key_quant, key_residual, value_quant, value_residual, kv_seq_len) = past_kv
        
        # Verify residual Growth
        k_res_len = key_residual.size(2) if key_residual is not None else 0
        v_res_len = value_residual.size(2) if value_residual is not None else 0
        
        print(f"  Key residual len: {k_res_len}")
        print(f"  Val residual len: {v_res_len}")
        print(f"  Quantized keys present: {key_quant is not None}")
        print(f"  Quantized vals present: {value_quant is not None}")
        
        # CHECK 1: Key Flush Logic (Batch flush when == R)
        if (t + 1) % residual_length == 0:
            # Just flushed? Residual should be empty (None) OR 0 if we handle it that way
            # My logic: if key_residual.size(2) == residual_length -> flush -> key_residual = None
            if key_residual is not None:
                print(f"  WARNING: Key residual should be None after flush at t={t}")
            else:
                print(f"  SUCCESS: Key flushed at t={t}")
        else:
            # Should be accumulating
            expected_len = (t + 1) % residual_length
            if k_res_len != expected_len:
                 print(f"  ERROR: Key residual len {k_res_len} != expected {expected_len}")

        # CHECK 2: Value Flush Logic (FIFO pop when > R)
        # Values grow up to R, then stay at R
        expected_v_len = min(t + 1, residual_length)
        # My logic: if value_residual.size(2) > residual_length -> pop 1
        # So it might temporarily be R+1 inside the function, but returned as R
        if v_res_len != expected_v_len:
             print(f"  ERROR: Val residual len {v_res_len} != expected {expected_v_len}")

    print("\nTest Complete")

if __name__ == "__main__":
    with torch.no_grad():
        test_kivi_streaming_logic()
