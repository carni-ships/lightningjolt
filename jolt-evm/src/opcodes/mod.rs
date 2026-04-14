//! EVM opcode handlers for trace verification
//!
//! Each opcode handler verifies the correctness of an execution step.

use crate::evm::{EvmStep, Opcode, U256};

/// Verify a single EVM step against its expected result
///
/// Returns `Ok(PostState)` if verification passes, `Err(VerificationError)` otherwise.
pub fn verify_step(step: &EvmStep, pre_state: &PreState) -> Result<PostState, VerificationError> {
    // Basic gas check
    if pre_state.gas < step.gas_used {
        return Err(VerificationError::GasExceeded);
    }

    match Opcode::from_u8(step.opcode) {
        Some(Opcode::STOP) => verify_stop(step, pre_state),
        Some(Opcode::ADD) => verify_add(step, pre_state),
        Some(Opcode::MUL) => verify_mul(step, pre_state),
        Some(Opcode::SUB) => verify_sub(step, pre_state),
        Some(Opcode::DIV) => verify_div(step, pre_state),
        Some(Opcode::SDIV) => verify_sdiv(step, pre_state),
        Some(Opcode::MOD) => verify_mod(step, pre_state),
        Some(Opcode::SMOD) => verify_smod(step, pre_state),
        Some(Opcode::ADDMOD) => verify_addmod(step, pre_state),
        Some(Opcode::MULMOD) => verify_mulmod(step, pre_state),
        Some(Opcode::EXP) => verify_exp(step, pre_state),
        Some(Opcode::SIGNEXTEND) => verify_signextend(step, pre_state),
        Some(Opcode::LT) => verify_lt(step, pre_state),
        Some(Opcode::GT) => verify_gt(step, pre_state),
        Some(Opcode::SLT) => verify_slt(step, pre_state),
        Some(Opcode::SGT) => verify_sgt(step, pre_state),
        Some(Opcode::EQ) => verify_eq(step, pre_state),
        Some(Opcode::ISZERO) => verify_iszero(step, pre_state),
        Some(Opcode::AND) => verify_and(step, pre_state),
        Some(Opcode::OR) => verify_or(step, pre_state),
        Some(Opcode::XOR) => verify_xor(step, pre_state),
        Some(Opcode::NOT) => verify_not(step, pre_state),
        Some(Opcode::BYTE) => verify_byte(step, pre_state),
        Some(Opcode::SHL) => verify_shl(step, pre_state),
        Some(Opcode::SHR) => verify_shr(step, pre_state),
        Some(Opcode::SAR) => verify_sar(step, pre_state),
        Some(Opcode::SHA3) => verify_sha3(step, pre_state),
        Some(Opcode::MLOAD) => verify_mload(step, pre_state),
        Some(Opcode::MSTORE) => verify_mstore(step, pre_state),
        Some(Opcode::MSTORE8) => verify_mstore8(step, pre_state),
        Some(Opcode::SLOAD) => verify_sload(step, pre_state),
        Some(Opcode::SSTORE) => verify_sstore(step, pre_state),
        Some(Opcode::JUMP) => verify_jump(step, pre_state),
        Some(Opcode::JUMPI) => verify_jumpi(step, pre_state),
        Some(Opcode::PC) => verify_pc(step, pre_state),
        Some(Opcode::JUMPDEST) => verify_jumpdest(step, pre_state),
        Some(Opcode::PUSH1) => verify_push(step, pre_state, 1),
        Some(Opcode::DUP1) => verify_dup(step, pre_state, 1),
        Some(Opcode::SWAP1) => verify_swap(step, pre_state, 1),
        Some(Opcode::RETURN) => verify_return(step, pre_state),
        Some(Opcode::REVERT) => verify_revert(step, pre_state),
        Some(Opcode::INVALID) => verify_invalid(step, pre_state),
        _ => Err(VerificationError::UnknownOpcode(step.opcode)),
    }
}

/// Pre-state before an EVM step
#[derive(Debug, Clone)]
pub struct PreState {
    pub pc: u64,
    pub gas: u64,
    pub sp: usize,
    pub stack: alloc::vec::Vec<U256>,
    pub memory: alloc::vec::Vec<u8>,
    pub storage: alloc::vec::Vec<(crate::evm::H256, crate::evm::H256)>,
}

/// Post-state after an EVM step
#[derive(Debug, Clone)]
pub struct PostState {
    pub pc: u64,
    pub gas: u64,
    pub sp: usize,
    pub stack: alloc::vec::Vec<U256>,
    pub memory: alloc::vec::Vec<u8>,
    pub storage: alloc::vec::Vec<(crate::evm::H256, crate::evm::H256)>,
    pub return_data: Option<(u64, u64)>,
}

/// Verification error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationError {
    UnknownOpcode(u8),
    StackUnderflow,
    StackOverflow,
    InvalidJumpDest,
    GasExceeded,
    MemoryOverflow,
    Revert,
    Invalid,
    Custom(&'static str),
}

// ============ Arithmetic Operations ============

fn verify_stop(_step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if !pre.stack.is_empty() {
        return Err(VerificationError::Custom("STOP should have empty stack"));
    }
    Ok(PostState {
        pc: pre.pc + 1,
        gas: pre.gas,
        sp: pre.sp,
        stack: pre.stack.clone(),
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_add(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("ADD stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = add_u256(a, b);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_mul(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("MUL stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = mul_u256(a, b);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_sub(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("SUB stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = sub_u256(a, b);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_div(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("DIV stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = div_u256(a, b);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_sdiv(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("SDIV stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = sdiv_u256(a, b);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_mod(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("MOD stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = mod_u256(a, b);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_smod(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("SMOD stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = smod_u256(a, b);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_addmod(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 2 {
        return Err(VerificationError::Custom("ADDMOD stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 3];
    let b = &pre.stack[pre.stack.len() - 2];
    let m = &pre.stack[pre.stack.len() - 1];
    let result = addmod_u256(a, b, m);
    let mut new_stack = pre.stack[..pre.stack.len() - 3].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_mulmod(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 2 {
        return Err(VerificationError::Custom("MULMOD stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 3];
    let b = &pre.stack[pre.stack.len() - 2];
    let m = &pre.stack[pre.stack.len() - 1];
    let result = mulmod_u256(a, b, m);
    let mut new_stack = pre.stack[..pre.stack.len() - 3].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_exp(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("EXP stack len mismatch"));
    }
    let base = &pre.stack[pre.stack.len() - 2];
    let exp = &pre.stack[pre.stack.len() - 1];
    let result = exp_u256(base, exp);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_signextend(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("SIGNEXTEND stack len mismatch"));
    }
    let b = &pre.stack[pre.stack.len() - 2];
    let x = &pre.stack[pre.stack.len() - 1];
    let result = signextend_u256(b, x);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

// ============ Comparison & Bitwise ============

fn verify_lt(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("LT stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = if lt_u256(a, b) { U256([1u8; 32]) } else { U256::zero() };
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_gt(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("GT stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = if gt_u256(a, b) { U256([1u8; 32]) } else { U256::zero() };
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_slt(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("SLT stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = if slt_u256(a, b) { U256([1u8; 32]) } else { U256::zero() };
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_sgt(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("SGT stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = if sgt_u256(a, b) { U256([1u8; 32]) } else { U256::zero() };
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_eq(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("EQ stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = if eq_u256(a, b) { U256([1u8; 32]) } else { U256::zero() };
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_iszero(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() {
        return Err(VerificationError::Custom("ISZERO stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 1];
    let result = if a.is_zero() { U256([1u8; 32]) } else { U256::zero() };
    let mut new_stack = pre.stack.clone();
    let idx = new_stack.len() - 1;
    new_stack[idx] = result;
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_and(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("AND stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = and_u256(a, b);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_or(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("OR stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = or_u256(a, b);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_xor(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("XOR stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 2];
    let b = &pre.stack[pre.stack.len() - 1];
    let result = xor_u256(a, b);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_not(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() {
        return Err(VerificationError::Custom("NOT stack len mismatch"));
    }
    let a = &pre.stack[pre.stack.len() - 1];
    let result = not_u256(a);
    let mut new_stack = pre.stack.clone();
    let idx = new_stack.len() - 1;
    new_stack[idx] = result;
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_byte(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("BYTE stack len mismatch"));
    }
    let pos = &pre.stack[pre.stack.len() - 2];
    let value = &pre.stack[pre.stack.len() - 1];
    let result = byte_u256(pos, value);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_shl(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("SHL stack len mismatch"));
    }
    let shift = &pre.stack[pre.stack.len() - 2];
    let value = &pre.stack[pre.stack.len() - 1];
    let result = shl_u256(value, shift);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_shr(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("SHR stack len mismatch"));
    }
    let shift = &pre.stack[pre.stack.len() - 2];
    let value = &pre.stack[pre.stack.len() - 1];
    let result = shr_u256(value, shift);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_sar(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("SAR stack len mismatch"));
    }
    let shift = &pre.stack[pre.stack.len() - 2];
    let value = &pre.stack[pre.stack.len() - 1];
    let result = sar_u256(value, shift);
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

// ============ SHA3 ============

fn verify_sha3(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 2 {
        return Err(VerificationError::Custom("SHA3 stack len mismatch"));
    }
    // SHA3 result is always 32 bytes (keccak256)
    // The result should be in step.stack.last()
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    if let Some(result) = step.stack.last() {
        new_stack.push(*result);
    }
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

// ============ Memory ============

fn verify_mload(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() {
        return Err(VerificationError::Custom("MLOAD stack len mismatch"));
    }
    // Result should be in step.stack.last() after MLOAD
    let mut new_stack = pre.stack.clone();
    if let Some(result) = step.stack.last() {
        let idx = new_stack.len() - 1;
        new_stack[idx] = *result;
    }
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_mstore(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 2 {
        return Err(VerificationError::Custom("MSTORE stack len mismatch"));
    }
    // MSTORE doesn't push to stack
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: pre.stack.clone(),
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_mstore8(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 2 {
        return Err(VerificationError::Custom("MSTORE8 stack len mismatch"));
    }
    // MSTORE8 doesn't push to stack
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: pre.stack.clone(),
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

// ============ Storage ============

fn verify_sload(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() {
        return Err(VerificationError::Custom("SLOAD stack len mismatch"));
    }
    // Result should be in step.stack.last() after SLOAD
    let mut new_stack = pre.stack.clone();
    if let Some(result) = step.stack.last() {
        let idx = new_stack.len() - 1;
        new_stack[idx] = *result;
    }
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_sstore(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 2 {
        return Err(VerificationError::Custom("SSTORE stack len mismatch"));
    }
    // SSTORE doesn't push to stack
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: pre.stack.clone(),
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

// ============ Control Flow ============

fn verify_jump(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 1 {
        return Err(VerificationError::Custom("JUMP stack len mismatch"));
    }
    let dest = &pre.stack[pre.stack.len() - 1];
    let new_pc = u256_to_u64(dest);
    Ok(PostState {
        pc: new_pc,
        gas: step.gas,
        sp: pre.sp,
        stack: pre.stack[..pre.stack.len() - 1].to_vec(),
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_jumpi(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 2 {
        return Err(VerificationError::Custom("JUMPI stack len mismatch"));
    }
    let dest = &pre.stack[pre.stack.len() - 2];
    let cond = &pre.stack[pre.stack.len() - 1];
    let new_pc = if !cond.is_zero() {
        u256_to_u64(dest)
    } else {
        step.pc + 1
    };
    let mut new_stack = pre.stack[..pre.stack.len() - 2].to_vec();
    Ok(PostState {
        pc: new_pc,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_pc(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() + 1 {
        return Err(VerificationError::Custom("PC stack len mismatch"));
    }
    let result = u64_to_u256(step.pc);
    let mut new_stack = pre.stack.clone();
    new_stack.push(result);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_jumpdest(_step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    Ok(PostState {
        pc: pre.pc + 1,
        gas: pre.gas,
        sp: pre.sp,
        stack: pre.stack.clone(),
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_push(step: &EvmStep, pre: &PreState, n: usize) -> Result<PostState, VerificationError> {
    // PUSH reads n+1 bytes (opcode + data), but stack grows by 1
    if step.stack.len() != pre.stack.len() + 1 {
        return Err(VerificationError::Custom("PUSH stack len mismatch"));
    }
    Ok(PostState {
        pc: step.pc + 1 + n as u64,
        gas: step.gas,
        sp: pre.sp + 1,
        stack: pre.stack.clone(),
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_dup(step: &EvmStep, pre: &PreState, n: usize) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() + 1 {
        return Err(VerificationError::Custom("DUP stack len mismatch"));
    }
    if pre.stack.len() < n {
        return Err(VerificationError::StackUnderflow);
    }
    let value = pre.stack[pre.stack.len() - n];
    let mut new_stack = pre.stack.clone();
    new_stack.push(value);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp + 1,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_swap(step: &EvmStep, pre: &PreState, n: usize) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() {
        return Err(VerificationError::Custom("SWAP stack len mismatch"));
    }
    if pre.stack.len() <= n {
        return Err(VerificationError::StackUnderflow);
    }
    let mut new_stack = pre.stack.clone();
    let len = new_stack.len();
    // Swap top with element at depth n+1 (1-indexed from top)
    // SWAP1 swaps positions len-1 and len-2
    // SWAPn swaps positions len-1 and len-n-1
    let idx1 = len - 1;
    let idx2 = len - 1 - n;
    new_stack.swap(idx1, idx2);
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: new_stack,
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: None,
    })
}

fn verify_return(step: &EvmStep, pre: &PreState) -> Result<PostState, VerificationError> {
    if step.stack.len() != pre.stack.len() - 2 {
        return Err(VerificationError::Custom("RETURN stack len mismatch"));
    }
    let offset = &pre.stack[pre.stack.len() - 2];
    let size = &pre.stack[pre.stack.len() - 1];
    Ok(PostState {
        pc: step.pc + 1,
        gas: step.gas,
        sp: pre.sp,
        stack: pre.stack[..pre.stack.len() - 2].to_vec(),
        memory: pre.memory.clone(),
        storage: pre.storage.clone(),
        return_data: Some((u256_to_u64(offset), u256_to_u64(size))),
    })
}

fn verify_revert(_step: &EvmStep, _pre: &PreState) -> Result<PostState, VerificationError> {
    Err(VerificationError::Revert)
}

fn verify_invalid(_step: &EvmStep, _pre: &PreState) -> Result<PostState, VerificationError> {
    Err(VerificationError::Invalid)
}

// ============ Helper Functions ============

fn add_u256(a: &U256, b: &U256) -> U256 {
    let mut result = U256::zero();
    let mut carry = 0u32;
    for i in 0..32 {
        let sum = (a.0[31 - i] as u32) + (b.0[31 - i] as u32) + carry;
        result.0[31 - i] = sum as u8;
        carry = sum >> 8;
    }
    result
}

fn sub_u256(a: &U256, b: &U256) -> U256 {
    let mut result = U256::zero();
    let mut borrow = 0i32;
    for i in 0..32 {
        let diff = (a.0[31 - i] as i32) - (b.0[31 - i] as i32) - borrow;
        if diff >= 0 {
            result.0[31 - i] = diff as u8;
            borrow = 0;
        } else {
            result.0[31 - i] = (diff + 256) as u8;
            borrow = 1;
        }
    }
    result
}

fn mul_u256(a: &U256, b: &U256) -> U256 {
    let mut result = U256::zero();
    for i in 0..32 {
        let mut carry: u32 = 0;
        for j in 0..(32 - i) {
            let product = (result.0[31 - i - j] as u32)
                + (a.0[31 - i] as u32) * (b.0[31 - j] as u32)
                + carry;
            result.0[31 - i - j] = product as u8;
            carry = product >> 8;
        }
    }
    result
}

fn div_u256(a: &U256, b: &U256) -> U256 {
    if b.is_zero() {
        return U256::zero();
    }
    // Simple division for small numbers
    let a_val = u256_to_u64(a);
    let b_val = u256_to_u64(b);
    if a_val < b_val {
        return U256::zero();
    }
    let result_val = a_val / b_val;
    u64_to_u256(result_val)
}

fn mod_u256(a: &U256, b: &U256) -> U256 {
    if b.is_zero() {
        return U256::zero();
    }
    let a_val = u256_to_u64(a);
    let b_val = u256_to_u64(b);
    let result_val = a_val % b_val;
    u64_to_u256(result_val)
}

fn sdiv_u256(a: &U256, b: &U256) -> U256 {
    // Signed division - simplified implementation
    if b.is_zero() {
        return U256::zero();
    }
    let a_sign = a.0[0] & 0x80 != 0;
    let b_sign = b.0[0] & 0x80 != 0;
    let a_abs = if a_sign { neg_u256(a) } else { *a };
    let b_abs = if b_sign { neg_u256(b) } else { *b };
    let a_val = u256_to_u64(&a_abs);
    let b_val = u256_to_u64(&b_abs);
    let result_val = a_val / b_val;
    let mut result = u64_to_u256(result_val);
    if a_sign != b_sign {
        result = neg_u256(&result);
    }
    result
}

fn smod_u256(a: &U256, b: &U256) -> U256 {
    if b.is_zero() {
        return U256::zero();
    }
    let a_sign = a.0[0] & 0x80 != 0;
    let a_abs = if a_sign { neg_u256(a) } else { *a };
    let b_val = u256_to_u64(b);
    let result_val = u256_to_u64(&a_abs) % b_val;
    let mut result = u64_to_u256(result_val);
    if a_sign {
        result = neg_u256(&result);
    }
    result
}

fn addmod_u256(a: &U256, b: &U256, m: &U256) -> U256 {
    if m.is_zero() {
        return U256::zero();
    }
    let a_val = u256_to_u64(a);
    let b_val = u256_to_u64(b);
    let m_val = u256_to_u64(m);
    let result_val = (a_val + b_val) % m_val;
    u64_to_u256(result_val)
}

fn mulmod_u256(a: &U256, b: &U256, m: &U256) -> U256 {
    if m.is_zero() {
        return U256::zero();
    }
    let a_val = u256_to_u64(a);
    let b_val = u256_to_u64(b);
    let m_val = u256_to_u64(m);
    let result_val = (a_val * b_val) % m_val;
    u64_to_u256(result_val)
}

fn exp_u256(base: &U256, exp: &U256) -> U256 {
    let base_val = u256_to_u64(base);
    let exp_val = u256_to_u64(exp) as u32;
    let result_val = base_val.pow(exp_val);
    u64_to_u256(result_val)
}

fn signextend_u256(b: &U256, x: &U256) -> U256 {
    let byte_idx = u256_to_u64(b) as usize;
    if byte_idx >= 31 {
        return *x;
    }
    let mut result = *x;
    let bit_idx = (byte_idx + 1) * 8 - 1;
    let mask = 1u8 << (bit_idx % 8);
    let sign_bit = result.0[byte_idx] & mask != 0;
    for i in byte_idx..32 {
        result.0[i] = if sign_bit { 0xff } else { 0 };
    }
    result
}

fn neg_u256(a: &U256) -> U256 {
    let mut result = U256::zero();
    let mut carry = 1u32;
    for i in 0..32 {
        let sum = (255 - a.0[31 - i] as u32) + carry;
        result.0[31 - i] = sum as u8;
        carry = sum >> 8;
    }
    result
}

fn lt_u256(a: &U256, b: &U256) -> bool {
    for i in 0..32 {
        if a.0[i] < b.0[i] {
            return true;
        }
        if a.0[i] > b.0[i] {
            return false;
        }
    }
    false
}

fn gt_u256(a: &U256, b: &U256) -> bool {
    for i in 0..32 {
        if a.0[i] > b.0[i] {
            return true;
        }
        if a.0[i] < b.0[i] {
            return false;
        }
    }
    false
}

fn slt_u256(a: &U256, b: &U256) -> bool {
    // Signed less than
    let a_sign = a.0[0] & 0x80 != 0;
    let b_sign = b.0[0] & 0x80 != 0;
    if a_sign != b_sign {
        return a_sign;
    }
    lt_u256(a, b)
}

fn sgt_u256(a: &U256, b: &U256) -> bool {
    // Signed greater than
    let a_sign = a.0[0] & 0x80 != 0;
    let b_sign = b.0[0] & 0x80 != 0;
    if a_sign != b_sign {
        return b_sign;
    }
    gt_u256(a, b)
}

fn eq_u256(a: &U256, b: &U256) -> bool {
    a.0 == b.0
}

fn and_u256(a: &U256, b: &U256) -> U256 {
    let mut result = U256::zero();
    for i in 0..32 {
        result.0[i] = a.0[i] & b.0[i];
    }
    result
}

fn or_u256(a: &U256, b: &U256) -> U256 {
    let mut result = U256::zero();
    for i in 0..32 {
        result.0[i] = a.0[i] | b.0[i];
    }
    result
}

fn xor_u256(a: &U256, b: &U256) -> U256 {
    let mut result = U256::zero();
    for i in 0..32 {
        result.0[i] = a.0[i] ^ b.0[i];
    }
    result
}

fn not_u256(a: &U256) -> U256 {
    let mut result = U256::zero();
    for i in 0..32 {
        result.0[i] = !a.0[i];
    }
    result
}

fn byte_u256(pos: &U256, value: &U256) -> U256 {
    let pos_val = u256_to_u64(pos) as usize;
    if pos_val >= 32 {
        return U256::zero();
    }
    let byte = value.0[pos_val];
    let mut result = U256::zero();
    result.0[31] = byte;
    result
}

fn shl_u256(value: &U256, shift: &U256) -> U256 {
    let mut result = U256::zero();
    let shift_bytes = shift.0[31];
    let byte_shift = (shift_bytes / 8) as usize;
    let bit_shift = shift_bytes % 8;

    for i in 0usize..32 {
        let src_idx = i.saturating_add(byte_shift);
        if src_idx < 32 {
            result.0[31 - i] = if bit_shift > 0 && src_idx > 0 {
                (value.0[31 - src_idx] << bit_shift)
                    | (value.0[31 - src_idx + 1] >> (8 - bit_shift))
            } else if bit_shift > 0 {
                value.0[31 - src_idx] << bit_shift
            } else {
                value.0[31 - src_idx]
            };
        }
    }
    result
}

fn shr_u256(value: &U256, shift: &U256) -> U256 {
    let mut result = U256::zero();
    let shift_bytes = shift.0[31];
    let byte_shift = (shift_bytes / 8) as usize;
    let bit_shift = shift_bytes % 8;

    for i in 0usize..32 {
        let src_idx = i.saturating_add(byte_shift);
        if src_idx < 32 {
            result.0[31 - i] = if bit_shift > 0 && src_idx < 31 {
                (value.0[31 - src_idx] >> bit_shift)
                    | (value.0[31 - src_idx - 1] << (8 - bit_shift))
            } else {
                value.0[31 - src_idx] >> bit_shift
            };
        }
    }
    result
}

fn sar_u256(value: &U256, shift: &U256) -> U256 {
    // Arithmetic right shift - preserve sign bit
    let sign_bit = value.0[0] & 0x80 != 0;
    let shift_bytes = shift.0[31];
    let byte_shift = (shift_bytes / 8) as usize;
    let bit_shift = shift_bytes % 8;

    let mut result = shr_u256(value, shift);

    if sign_bit && byte_shift < 32 {
        // Fill in high bytes with sign bit
        for i in 0..byte_shift.min(32) {
            result.0[i] = 0xff;
        }
        if bit_shift > 0 && byte_shift < 31 {
            let mask = !(0xff >> bit_shift);
            result.0[byte_shift] |= mask;
        }
    }
    result
}

fn u256_to_u64(a: &U256) -> u64 {
    let mut result = 0u64;
    for i in 0..8 {
        result = (result << 8) | (a.0[i] as u64);
    }
    result
}

fn u64_to_u256(a: u64) -> U256 {
    let mut result = U256::zero();
    for i in 0..8 {
        result.0[31 - i] = (a >> (56 - i * 8)) as u8;
    }
    result
}
