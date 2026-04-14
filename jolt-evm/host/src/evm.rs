//! Simple EVM Executor for trace generation
//!
//! This module implements a minimal EVM interpreter that produces
//! step-by-step execution traces for verification by the jolt-evm guest.

use jolt_evm::{Address, H256, U256, EvmStep, EvmTrace, Opcode};
use std::vec::Vec;

/// Gas constants (per EVM specification)
const GAS_BASE: u64 = 2;
const GAS_PUSH: u64 = 3;
const GAS_ADD: u64 = 3;
const GAS_MUL: u64 = 5;
const GAS_SUB: u64 = 3;
const GAS_DIV: u64 = 5;
const GAS_SDIV: u64 = 5;
const GAS_MOD: u64 = 5;
const GAS_SMOD: u64 = 5;
const GAS_ADDMOD: u64 = 8;
const GAS_MULMOD: u64 = 8;
const GAS_EXP: u64 = 10;
const GAS_SIGNEXTEND: u64 = 5;
const GAS_LT: u64 = 3;
const GAS_GT: u64 = 3;
const GAS_SLT: u64 = 3;
const GAS_SGT: u64 = 3;
const GAS_EQ: u64 = 3;
const GAS_ISZERO: u64 = 3;
const GAS_AND: u64 = 3;
const GAS_OR: u64 = 3;
const GAS_XOR: u64 = 3;
const GAS_NOT: u64 = 3;
const GAS_BYTE: u64 = 3;
const GAS_SHL: u64 = 3;
const GAS_SHR: u64 = 3;
const GAS_SAR: u64 = 3;
const GAS_SHA3: u64 = 30;
const GAS_MLOAD: u64 = 3;
const GAS_MSTORE: u64 = 3;
const GAS_MSTORE8: u64 = 3;
const GAS_SLOAD: u64 = 100;
const GAS_SSTORE: u64 = 100;
const GAS_JUMP: u64 = 8;
const GAS_JUMPI: u64 = 10;
const GAS_PC: u64 = 2;
const GAS_JUMPDEST: u64 = 1;

/// Simple stack for EVM execution
#[derive(Debug, Clone)]
struct Stack {
    data: Vec<U256>,
}

impl Stack {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn push(&mut self, value: U256) {
        self.data.push(value);
    }

    fn pop(&mut self) -> Option<U256> {
        self.data.pop()
    }

    fn dup(&self, n: usize) -> Option<U256> {
        if n > 0 && n <= self.data.len() {
            Some(self.data[self.data.len() - n])
        } else {
            None
        }
    }

    fn swap(&mut self, n: usize) {
        if n > 0 && n < self.data.len() {
            let len = self.data.len();
            self.data.swap(len - 1, len - 1 - n);
        }
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Simple memory for EVM execution
#[derive(Debug, Clone)]
struct Memory {
    data: Vec<u8>,
}

impl Memory {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn mload(&self, offset: u64) -> Option<u64> {
        if offset as usize + 8 <= self.data.len() {
            let mut val = 0u64;
            for i in 0..8 {
                val |= (self.data[offset as usize + i] as u64) << (i * 8);
            }
            Some(val)
        } else {
            Some(0)
        }
    }

    fn mstore(&mut self, offset: u64, value: u64) {
        let offset = offset as usize;
        if offset + 8 > self.data.len() {
            self.data.resize(offset + 8, 0);
        }
        for i in 0..8 {
            self.data[offset + i] = (value >> (i * 8)) as u8;
        }
    }

    fn mstore8(&mut self, offset: u64, value: u8) {
        let offset = offset as usize;
        if offset >= self.data.len() {
            self.data.resize(offset + 1, 0);
        }
        self.data[offset] = value;
    }
}

/// EVM Executor that produces traces
pub struct EvmExecutor {
    pc: u64,
    gas: u64,
    stack: Stack,
    memory: Memory,
    storage: Vec<(H256, H256)>,
    return_data: Vec<u8>,
}

impl EvmExecutor {
    pub fn new(gas_limit: u64) -> Self {
        Self {
            pc: 0,
            gas: gas_limit,
            stack: Stack::new(),
            memory: Memory::new(),
            storage: Vec::new(),
            return_data: Vec::new(),
        }
    }

    /// Execute bytecode and produce a trace
    pub fn execute(
        bytecode: &[u8],
        entry_pc: u64,
        caller: Address,
        callee: Address,
        value: U256,
        input_data: Vec<u8>,
        gas_limit: u64,
    ) -> Result<EvmTrace, &'static str> {
        let mut executor = Self::new(gas_limit);
        executor.pc = entry_pc;

        let code_hash = {
            use tiny_keccak::Hasher;
            let mut hasher = tiny_keccak::Keccak::v256();
            let mut output = [0u8; 32];
            hasher.update(bytecode);
            hasher.finalize(&mut output);
            H256(output)
        };

        let mut steps = Vec::new();
        let mut success = true;

        loop {
            if executor.pc as usize >= bytecode.len() {
                break;
            }

            let opcode = bytecode[executor.pc as usize];
            let opcode_fn = Opcode::from_u8(opcode);

            // Determine gas cost based on opcode
            let gas_cost = match opcode_fn {
                Some(Opcode::STOP) | Some(Opcode::REVERT) | Some(Opcode::INVALID) => 0,
                Some(Opcode::PUSH1) => GAS_PUSH, // PUSH1-PUSH32 all map to PUSH1
                Some(Opcode::ADD) => GAS_ADD,
                Some(Opcode::MUL) => GAS_MUL,
                Some(Opcode::SUB) => GAS_SUB,
                Some(Opcode::DIV) => GAS_DIV,
                Some(Opcode::SDIV) => GAS_SDIV,
                Some(Opcode::MOD) => GAS_MOD,
                Some(Opcode::SMOD) => GAS_SMOD,
                Some(Opcode::ADDMOD) => GAS_ADDMOD,
                Some(Opcode::MULMOD) => GAS_MULMOD,
                Some(Opcode::EXP) => GAS_EXP,
                Some(Opcode::SIGNEXTEND) => GAS_SIGNEXTEND,
                Some(Opcode::LT) => GAS_LT,
                Some(Opcode::GT) => GAS_GT,
                Some(Opcode::SLT) => GAS_SLT,
                Some(Opcode::SGT) => GAS_SGT,
                Some(Opcode::EQ) => GAS_EQ,
                Some(Opcode::ISZERO) => GAS_ISZERO,
                Some(Opcode::AND) => GAS_AND,
                Some(Opcode::OR) => GAS_OR,
                Some(Opcode::XOR) => GAS_XOR,
                Some(Opcode::NOT) => GAS_NOT,
                Some(Opcode::BYTE) => GAS_BYTE,
                Some(Opcode::SHL) => GAS_SHL,
                Some(Opcode::SHR) => GAS_SHR,
                Some(Opcode::SAR) => GAS_SAR,
                Some(Opcode::SHA3) => GAS_SHA3,
                Some(Opcode::MLOAD) => GAS_MLOAD,
                Some(Opcode::MSTORE) => GAS_MSTORE,
                Some(Opcode::MSTORE8) => GAS_MSTORE8,
                Some(Opcode::SLOAD) => GAS_SLOAD,
                Some(Opcode::SSTORE) => GAS_SSTORE,
                Some(Opcode::JUMP) => GAS_JUMP,
                Some(Opcode::JUMPI) => GAS_JUMPI,
                Some(Opcode::PC) => GAS_PC,
                Some(Opcode::JUMPDEST) => GAS_JUMPDEST,
                Some(Opcode::DUP1) => GAS_PUSH, // DUP1-DUP16 all map to DUP1
                Some(Opcode::SWAP1) => GAS_PUSH, // SWAP1-SWAP16 all map to SWAP1
                _ => GAS_BASE,
            };

            // Check gas
            if executor.gas < gas_cost {
                success = false;
                break;
            }

            let gas_before = executor.gas;
            executor.gas -= gas_cost;

            // Create step before execution
            let step = Self::make_step(
                executor.pc,
                opcode,
                gas_before,
                gas_cost,
                &executor.stack,
                &executor.memory,
            );

            // Execute the opcode
            match Self::execute_opcode(&mut executor, bytecode, opcode) {
                Ok(()) => {}
                Err(_) => {
                    success = false;
                    break;
                }
            }

            steps.push(step);

            // Handle control flow opcodes
            match opcode_fn {
                Some(Opcode::STOP) | Some(Opcode::RETURN) | Some(Opcode::REVERT) => {
                    break;
                }
                _ => {}
            }
        }

        Ok(EvmTrace {
            entry_pc,
            code_hash,
            caller,
            callee,
            value,
            input_data,
            steps,
            final_gas: executor.gas,
            success,
            return_data: executor.return_data,
        })
    }

    fn make_step(
        pc: u64,
        opcode: u8,
        gas: u64,
        gas_used: u64,
        stack: &Stack,
        _memory: &Memory,
    ) -> EvmStep {
        EvmStep {
            pc,
            opcode,
            gas,
            gas_used,
            stack: stack.data.clone(),
            memory: None,
            storage_key: None,
            storage_value_before: None,
            storage_value_after: None,
            return_data: None,
        }
    }

    fn execute_opcode(&mut self, bytecode: &[u8], opcode: u8) -> Result<(), &'static str> {
        match Opcode::from_u8(opcode) {
            Some(Opcode::STOP) => {}
            Some(Opcode::ADD) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::add_u256(a, b);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::MUL) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::mul_u256(a, b);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::SUB) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::sub_u256(a, b);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::DIV) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::div_u256(a, b);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::SDIV) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::sdiv_u256(a, b);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::MOD) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::mod_u256(a, b);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::SMOD) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::smod_u256(a, b);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::ADDMOD) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = self.stack.pop().ok_or("stack underflow")?;
                let result = Self::addmod_u256(a, b, c);
                self.stack.push(result);
                self.pc += 1;
            }
            Some(Opcode::MULMOD) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = self.stack.pop().ok_or("stack underflow")?;
                let result = Self::mulmod_u256(a, b, c);
                self.stack.push(result);
                self.pc += 1;
            }
            Some(Opcode::EXP) => {
                let base = self.stack.pop().ok_or("stack underflow")?;
                let exp = self.stack.pop().ok_or("stack underflow")?;
                let result = Self::exp_u256(base, exp);
                self.stack.push(result);
                self.pc += 1;
            }
            Some(Opcode::SIGNEXTEND) => {
                let size = self.stack.pop().ok_or("stack underflow")?;
                let value = self.stack.pop().ok_or("stack underflow")?;
                let result = Self::signextend_u256(value, size);
                self.stack.push(result);
                self.pc += 1;
            }
            Some(Opcode::LT) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = if Self::lt_u256(a, b) { Self::u256_one() } else { Self::u256_zero() };
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::GT) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = if Self::gt_u256(a, b) { Self::u256_one() } else { Self::u256_zero() };
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::SLT) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = if Self::slt_u256(a, b) { Self::u256_one() } else { Self::u256_zero() };
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::SGT) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = if Self::sgt_u256(a, b) { Self::u256_one() } else { Self::u256_zero() };
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::EQ) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = if a == b { Self::u256_one() } else { Self::u256_zero() };
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::ISZERO) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let c = if a.0.iter().all(|&x| x == 0) { Self::u256_one() } else { Self::u256_zero() };
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::AND) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::and_u256(a, b);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::OR) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::or_u256(a, b);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::XOR) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let b = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::xor_u256(a, b);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::NOT) => {
                let a = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::not_u256(a);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::BYTE) => {
                let offset = self.stack.pop().ok_or("stack underflow")?;
                let value = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::byte_u256(offset, value);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::SHL) => {
                let shift = self.stack.pop().ok_or("stack underflow")?;
                let value = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::shl_u256(value, shift);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::SHR) => {
                let shift = self.stack.pop().ok_or("stack underflow")?;
                let value = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::shr_u256(value, shift);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::SAR) => {
                let shift = self.stack.pop().ok_or("stack underflow")?;
                let value = self.stack.pop().ok_or("stack underflow")?;
                let c = Self::sar_u256(value, shift);
                self.stack.push(c);
                self.pc += 1;
            }
            Some(Opcode::PUSH1) => {
                // PUSH1 = 0x60, PUSH2 = 0x61, ..., PUSH32 = 0x7f
                // All map to Opcode::PUSH1, detect actual length from opcode byte
                let push_opcode = bytecode.get(self.pc as usize).unwrap_or(&0x60);
                let push_len = (*push_opcode as usize).saturating_sub(0x5f).min(32);
                let mut bytes = [0u8; 32];
                for i in 0..push_len {
                    if self.pc as usize + 1 + i < bytecode.len() {
                        bytes[i] = bytecode[self.pc as usize + 1 + i];
                    }
                }
                self.stack.push(U256(bytes));
                self.pc += push_len as u64 + 1;
            }
            Some(Opcode::DUP1) => {
                // DUP1 = 0x80, DUP2 = 0x81, ..., DUP16 = 0x8f
                // All map to Opcode::DUP1, detect actual n from opcode byte
                let dup_opcode = bytecode.get(self.pc as usize).unwrap_or(&0x80);
                let dup_n = (*dup_opcode as usize).saturating_sub(0x7f).min(16);
                if let Some(val) = self.stack.dup(dup_n) {
                    self.stack.push(val);
                }
                self.pc += 1;
            }
            Some(Opcode::SWAP1) => {
                // SWAP1 = 0x90, SWAP2 = 0x91, ..., SWAP16 = 0x9f
                // All map to Opcode::SWAP1, detect actual n from opcode byte
                let swap_opcode = bytecode.get(self.pc as usize).unwrap_or(&0x90);
                let swap_n = (*swap_opcode as usize).saturating_sub(0x8f).min(16);
                self.stack.swap(swap_n);
                self.pc += 1;
            }
            Some(Opcode::JUMP) => {
                let dest = self.stack.pop().ok_or("stack underflow")?;
                let dest_pc = Self::u256_to_u64(dest);
                // Verify jumpdest is valid (bytecode[dest_pc] == JUMPDEST)
                if bytecode.len() > dest_pc as usize && bytecode[dest_pc as usize] == 0x5b {
                    self.pc = dest_pc;
                } else {
                    return Err("invalid jump dest");
                }
            }
            Some(Opcode::JUMPI) => {
                let dest = self.stack.pop().ok_or("stack underflow")?;
                let cond = self.stack.pop().ok_or("stack underflow")?;
                if !cond.0.iter().all(|&x| x == 0) {
                    let dest_pc = Self::u256_to_u64(dest);
                    if bytecode.len() > dest_pc as usize && bytecode[dest_pc as usize] == 0x5b {
                        self.pc = dest_pc;
                    } else {
                        return Err("invalid jump dest");
                    }
                } else {
                    self.pc += 1;
                }
            }
            Some(Opcode::JUMPDEST) => {
                self.pc += 1;
            }
            Some(Opcode::PC) => {
                self.stack.push(Self::u256_from_u64(self.pc));
                self.pc += 1;
            }
            Some(Opcode::MLOAD) => {
                let offset = self.stack.pop().ok_or("stack underflow")?;
                let offset_u64 = Self::u256_to_u64(offset);
                let val = self.memory.mload(offset_u64).unwrap_or(0);
                self.stack.push(Self::u256_from_u64(val));
                self.pc += 1;
            }
            Some(Opcode::MSTORE) => {
                let offset = self.stack.pop().ok_or("stack underflow")?;
                let value = self.stack.pop().ok_or("stack underflow")?;
                self.memory.mstore(Self::u256_to_u64(offset), Self::u256_to_u64(value));
                self.pc += 1;
            }
            Some(Opcode::MSTORE8) => {
                let offset = self.stack.pop().ok_or("stack underflow")?;
                let value = self.stack.pop().ok_or("stack underflow")?;
                self.memory.mstore8(Self::u256_to_u64(offset), Self::u256_to_u64(value) as u8);
                self.pc += 1;
            }
            Some(Opcode::SHA3) => {
                let offset = self.stack.pop().ok_or("stack underflow")?;
                let size = self.stack.pop().ok_or("stack underflow")?;
                let offset_u64 = Self::u256_to_u64(offset);
                let size_u64 = Self::u256_to_u64(size);
                let mut data = Vec::new();
                for i in 0..size_u64 {
                    if offset_u64 + i < self.memory.data.len() as u64 {
                        data.push(self.memory.data[(offset_u64 + i) as usize]);
                    }
                }
                let hash = Self::keccak256(&data);
                self.stack.push(U256(hash.0));
                self.pc += 1;
            }
            Some(Opcode::RETURN) => {
                let offset = self.stack.pop().ok_or("stack underflow")?;
                let size = self.stack.pop().ok_or("stack underflow")?;
                let offset_u64 = Self::u256_to_u64(offset);
                let size_u64 = Self::u256_to_u64(size);
                self.return_data.clear();
                for i in 0..size_u64 {
                    if offset_u64 + i < self.memory.data.len() as u64 {
                        self.return_data.push(self.memory.data[(offset_u64 + i) as usize]);
                    }
                }
            }
            Some(Opcode::REVERT) => {
                return Err("revert");
            }
            Some(Opcode::INVALID) => {
                return Err("invalid");
            }
            _ => {
                return Err("unknown opcode");
            }
        }
        Ok(())
    }

    // U256 helper functions
    fn u256_zero() -> U256 {
        U256([0u8; 32])
    }

    fn u256_one() -> U256 {
        let mut bytes = [0u8; 32];
        bytes[0] = 1;
        U256(bytes)
    }

    fn u256_from_u64(v: u64) -> U256 {
        let mut bytes = [0u8; 32];
        for (i, &byte) in v.to_le_bytes().iter().enumerate() {
            bytes[i] = byte;
        }
        U256(bytes)
    }

    fn u256_to_u64(v: U256) -> u64 {
        v.0[0] as u64
    }

    fn from_u256(v: U256) -> u64 {
        v.0[0] as u64
    }

    fn add_u256(a: U256, b: U256) -> U256 {
        let a = Self::from_u256(a);
        let b = Self::from_u256(b);
        let result = a.wrapping_add(b);
        Self::u256_from_u64(result)
    }

    fn sub_u256(a: U256, b: U256) -> U256 {
        let a = Self::from_u256(a);
        let b = Self::from_u256(b);
        let result = a.wrapping_sub(b);
        Self::u256_from_u64(result)
    }

    fn mul_u256(a: U256, b: U256) -> U256 {
        let a = Self::from_u256(a);
        let b = Self::from_u256(b);
        let result = a.wrapping_mul(b);
        Self::u256_from_u64(result)
    }

    fn div_u256(a: U256, b: U256) -> U256 {
        let a = Self::from_u256(a);
        let b = Self::from_u256(b);
        let result = if b == 0 { 0 } else { a / b };
        Self::u256_from_u64(result)
    }

    fn sdiv_u256(a: U256, b: U256) -> U256 {
        Self::div_u256(a, b)
    }

    fn mod_u256(a: U256, b: U256) -> U256 {
        let a = Self::from_u256(a);
        let b = Self::from_u256(b);
        let result = if b == 0 { 0 } else { a % b };
        Self::u256_from_u64(result)
    }

    fn smod_u256(a: U256, b: U256) -> U256 {
        Self::mod_u256(a, b)
    }

    fn addmod_u256(a: U256, b: U256, c: U256) -> U256 {
        let a = Self::from_u256(a);
        let b = Self::from_u256(b);
        let c = Self::from_u256(c);
        let result = if c == 0 { 0 } else { (a + b) % c };
        Self::u256_from_u64(result)
    }

    fn mulmod_u256(a: U256, b: U256, c: U256) -> U256 {
        let a = Self::from_u256(a);
        let b = Self::from_u256(b);
        let c = Self::from_u256(c);
        let result = if c == 0 { 0 } else { (a * b) % c };
        Self::u256_from_u64(result)
    }

    fn exp_u256(base: U256, exp: U256) -> U256 {
        let base = Self::from_u256(base);
        let exp = Self::from_u256(exp) as u32;
        let result = base.wrapping_pow(exp.into());
        Self::u256_from_u64(result)
    }

    fn signextend_u256(value: U256, size: U256) -> U256 {
        let size = Self::from_u256(size) as usize;
        let mut value_bytes = value.0;
        if size < 31 {
            let bit_pos = size * 8 + 7;
            let bit = if bit_pos < 256 { value_bytes[bit_pos / 8] >> (bit_pos % 8) & 1 } else { 0 };
            if bit == 1 {
                for i in 0..=size {
                    if i < 32 {
                        value_bytes[i] = 0xff;
                    }
                }
            }
        }
        U256(value_bytes)
    }

    fn lt_u256(a: U256, b: U256) -> bool {
        Self::from_u256(a) < Self::from_u256(b)
    }

    fn gt_u256(a: U256, b: U256) -> bool {
        Self::from_u256(a) > Self::from_u256(b)
    }

    fn slt_u256(a: U256, b: U256) -> bool {
        Self::lt_u256(a, b)
    }

    fn sgt_u256(a: U256, b: U256) -> bool {
        Self::gt_u256(a, b)
    }

    fn and_u256(a: U256, b: U256) -> U256 {
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = a.0[i] & b.0[i];
        }
        U256(result)
    }

    fn or_u256(a: U256, b: U256) -> U256 {
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = a.0[i] | b.0[i];
        }
        U256(result)
    }

    fn xor_u256(a: U256, b: U256) -> U256 {
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = a.0[i] ^ b.0[i];
        }
        U256(result)
    }

    fn not_u256(a: U256) -> U256 {
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = !a.0[i];
        }
        U256(result)
    }

    fn byte_u256(offset: U256, value: U256) -> U256 {
        let offset = Self::from_u256(offset) as usize;
        let mut result = [0u8; 32];
        if offset < 32 {
            result[0] = value.0[offset];
        }
        U256(result)
    }

    fn shl_u256(value: U256, shift: U256) -> U256 {
        let shift = Self::from_u256(shift);
        let value = Self::from_u256(value);
        let result = value << shift;
        Self::u256_from_u64(result)
    }

    fn shr_u256(value: U256, shift: U256) -> U256 {
        let shift = Self::from_u256(shift);
        let value = Self::from_u256(value);
        let result = value >> shift;
        Self::u256_from_u64(result)
    }

    fn sar_u256(value: U256, shift: U256) -> U256 {
        Self::shr_u256(value, shift)
    }

    fn keccak256(data: &[u8]) -> H256 {
        use tiny_keccak::Hasher;
        let mut hasher = tiny_keccak::Keccak::v256();
        let mut output = [0u8; 32];
        hasher.update(data);
        hasher.finalize(&mut output);
        H256(output)
    }
}

// Re-export for convenience
pub use EvmExecutor as Executor;
