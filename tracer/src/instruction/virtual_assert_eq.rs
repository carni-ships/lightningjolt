use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_b::FormatB, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAssertEQ,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = ()
);

impl VirtualAssertEQ {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualAssertEQ as RISCVInstruction>::RAMAccess) {
        let rs1_val = cpu.x[self.operands.rs1 as usize];
        let rs2_val = cpu.x[self.operands.rs2 as usize];
        if self.operands.imm == 0 {
            if rs1_val != rs2_val {
                let trigger_pc = cpu.pc.wrapping_sub(8);
                eprintln!("[VirtualAssertEQ FAIL] rs1={} (reg {}), rs2={} (reg {}), trigger_pc=0x{:x}",
                    rs1_val, self.operands.rs1, rs2_val, self.operands.rs2, trigger_pc);
            }
            assert_eq!(rs1_val, rs2_val);
        } else {
            if rs1_val != rs2_val {
                tracing::warn!(
                    "VirtualAssertEQ (spoil): rs1={rs1_val} != rs2={rs2_val}, proof will be unsatisfiable",
                );
            }
        }
    }
}

impl RISCVTrace for VirtualAssertEQ {}
