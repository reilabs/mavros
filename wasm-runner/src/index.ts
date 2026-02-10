#!/usr/bin/env node
/**
 * Mavros WASM Runner CLI
 */

import * as fs from 'fs';
import * as path from 'path';
import { parseInputs } from './input.js';
import { run, writeResult } from './runner.js';

function printUsage(): void {
  console.log(`
Usage: mavros-wasm-runner <project-root>

Arguments:
  project-root    Path to Noir project (default: ./)

Expected project structure:
  <project-root>/
    Prover.toml                      # Input values
    mavros_debug/
      witgen.wasm                    # Generated WASM artifact
      witgen.wasm.meta.json          # Metadata

Output is written to: <project-root>/mavros_debug/output.json
`);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.includes('-h') || args.includes('--help')) {
    printUsage();
    process.exit(0);
  }

  const projectRoot = args[0] || './';
  const inputPath = path.join(projectRoot, 'Prover.toml');
  const wasmPath = path.join(projectRoot, 'mavros_debug', 'witgen.wasm');
  const outputPath = path.join(projectRoot, 'mavros_debug', 'output.json');

  // Validate paths
  if (!fs.existsSync(projectRoot)) {
    console.error(`Error: Project root not found: ${projectRoot}`);
    process.exit(1);
  }
  if (!fs.existsSync(wasmPath)) {
    console.error(`Error: WASM file not found: ${wasmPath}`);
    console.error(`Run mavros --emit-wasm on the project first.`);
    process.exit(1);
  }
  if (!fs.existsSync(`${wasmPath}.meta.json`)) {
    console.error(`Error: Metadata not found: ${wasmPath}.meta.json`);
    process.exit(1);
  }
  if (!fs.existsSync(inputPath)) {
    console.error(`Error: Input file not found: ${inputPath}`);
    process.exit(1);
  }

  try {
    const { inputs, metadata } = parseInputs(wasmPath, inputPath);
    console.log(`Running ${wasmPath} with ${inputs.length} inputs...`);

    const start = performance.now();
    const result = await run(wasmPath, inputs, metadata);
    const elapsed = performance.now() - start;

    writeResult(result, outputPath);
    console.log(`Done in ${elapsed.toFixed(0)}ms`);
    console.log(`  WASM execution: ${result.executionTimeMs.toFixed(0)}ms`);
    console.log(`  Witnesses: ${result.witnesses.length}`);
    console.log(`  Constraints: ${result.constraints.a.length}`);
    console.log(`  Output: ${outputPath}`);
  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

main();
