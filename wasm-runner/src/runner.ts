/**
 * WASM Runner - executes generated WASM with native WASM runtime functions
 */

import * as fs from 'fs';
import * as path from 'path';
import { FieldElement, fieldToHex } from './field.js';
import { ProgramMetadata } from './input.js';

const FIELD_SIZE = 32; // 4 x i64 = 32 bytes

export interface RunResult {
  witnesses: string[];
  constraints: {
    a: string[];
    b: string[];
    c: string[];
  };
  executionTimeMs: number;
}

// Path to the compiled WASM runtime
const RUNTIME_WASM_PATH = path.join(
  path.dirname(new URL(import.meta.url).pathname),
  '../../target/wasm32-unknown-unknown/release/mavros_wasm_runtime.wasm'
);

let runtimeModule: WebAssembly.Module | null = null;

async function loadRuntimeModule(): Promise<WebAssembly.Module> {
  if (runtimeModule) return runtimeModule;
  const runtimeBytes = fs.readFileSync(RUNTIME_WASM_PATH);
  runtimeModule = await WebAssembly.compile(runtimeBytes);
  return runtimeModule;
}

/**
 * Read a field element from memory
 */
function readField(view: DataView, ptr: number): FieldElement {
  return [
    view.getBigUint64(ptr, true),
    view.getBigUint64(ptr + 8, true),
    view.getBigUint64(ptr + 16, true),
    view.getBigUint64(ptr + 24, true),
  ];
}

/**
 * Write a field element to memory
 */
function writeField(view: DataView, ptr: number, field: FieldElement): void {
  view.setBigUint64(ptr, field[0], true);
  view.setBigUint64(ptr + 8, field[1], true);
  view.setBigUint64(ptr + 16, field[2], true);
  view.setBigUint64(ptr + 24, field[3], true);
}

export async function run(
  wasmPath: string,
  inputs: FieldElement[],
  metadata: ProgramMetadata
): Promise<RunResult> {
  // Calculate memory size needed
  const witnessBytes = metadata.witnessCount * FIELD_SIZE;
  const constraintBytes = metadata.constraintCount * FIELD_SIZE;
  const totalBytes = 64 + witnessBytes + constraintBytes * 3; // VM struct + buffers
  const pages = Math.ceil(totalBytes / 65536) + 1;

  // Create shared memory
  const memory = new WebAssembly.Memory({ initial: pages, maximum: pages * 2 });
  const view = new DataView(memory.buffer);

  // Layout: VM struct at 0, then buffers
  const vmPtr = 0;
  const witnessPtr = 64;
  const aPtr = witnessPtr + witnessBytes;
  const bPtr = aPtr + constraintBytes;
  const cPtr = bPtr + constraintBytes;

  // Initialize VM struct with buffer pointers (these get advanced by write functions)
  view.setUint32(vmPtr, witnessPtr, true);
  view.setUint32(vmPtr + 4, aPtr, true);
  view.setUint32(vmPtr + 8, bPtr, true);
  view.setUint32(vmPtr + 12, cPtr, true);

  // Load the WASM runtime module (provides __field_mul)
  const runtimeMod = await loadRuntimeModule();
  const runtimeInstance = await WebAssembly.instantiate(runtimeMod, {
    env: { memory },
  });
  const runtimeExports = runtimeInstance.exports as {
    __field_mul: Function;
  };

  // Import __field_mul from the runtime WASM module
  const imports: WebAssembly.Imports = {
    env: {
      memory,
      __field_mul: runtimeExports.__field_mul,
    },
  };

  // Load and run the generated WASM
  const wasmBytes = fs.readFileSync(wasmPath);
  const module = await WebAssembly.compile(wasmBytes);
  const instance = await WebAssembly.instantiate(module, imports);
  const exports = instance.exports as { mavros_main: (...args: (number | bigint)[]) => void };

  // Prepare arguments: vmPtr, then flattened input limbs
  const args: (number | bigint)[] = [vmPtr];
  for (const input of inputs) {
    args.push(input[0], input[1], input[2], input[3]);
  }

  // Execute
  const execStart = performance.now();
  exports.mavros_main(...args);
  const executionTimeMs = performance.now() - execStart;

  // Read output buffers
  const witnesses: FieldElement[] = [];
  const constraintsA: FieldElement[] = [];
  const constraintsB: FieldElement[] = [];
  const constraintsC: FieldElement[] = [];

  for (let i = 0; i < metadata.witnessCount; i++) {
    witnesses.push(readField(view, witnessPtr + i * FIELD_SIZE));
  }
  for (let i = 0; i < metadata.constraintCount; i++) {
    constraintsA.push(readField(view, aPtr + i * FIELD_SIZE));
    constraintsB.push(readField(view, bPtr + i * FIELD_SIZE));
    constraintsC.push(readField(view, cPtr + i * FIELD_SIZE));
  }

  return {
    witnesses: witnesses.map(fieldToHex),
    constraints: {
      a: constraintsA.map(fieldToHex),
      b: constraintsB.map(fieldToHex),
      c: constraintsC.map(fieldToHex),
    },
    executionTimeMs,
  };
}

export function writeResult(result: RunResult, outputPath: string): void {
  const { executionTimeMs, ...output } = result;
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
}
