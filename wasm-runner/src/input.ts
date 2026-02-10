/**
 * Prover.toml Input Parsing
 *
 * Parses input values from Prover.toml and converts them to field elements.
 * Supports nested arrays and handles parameter ordering based on metadata.
 */

import * as fs from 'fs';
import * as TOML from 'toml';
import { FieldElement, parseFieldElement } from './field.js';

/**
 * Metadata about the compiled program
 */
export interface ProgramMetadata {
  witnessCount: number;
  constraintCount: number;
  parameters: ParameterInfo[];
}

/**
 * Information about a single parameter
 */
export interface ParameterInfo {
  name: string;
  // The total number of field elements this parameter occupies
  // (flattened for arrays)
  elementCount: number;
}

/**
 * Recursively flatten a value (which may be nested arrays) to field element strings
 */
function flattenValue(value: unknown): string[] {
  if (typeof value === 'string') {
    return [value];
  }

  if (typeof value === 'number') {
    return [value.toString()];
  }

  if (Array.isArray(value)) {
    const result: string[] = [];
    for (const item of value) {
      result.push(...flattenValue(item));
    }
    return result;
  }

  throw new Error(`Unsupported value type in Prover.toml: ${typeof value}`);
}

/**
 * Parse Prover.toml and return field elements in parameter order
 */
export function parseProverToml(
  tomlPath: string,
  metadata: ProgramMetadata
): FieldElement[] {
  // Read and parse TOML file
  const content = fs.readFileSync(tomlPath, 'utf-8');
  const parsed = TOML.parse(content) as Record<string, unknown>;

  const allElements: FieldElement[] = [];

  // Process parameters in the order specified by metadata
  for (const param of metadata.parameters) {
    const value = parsed[param.name];

    if (value === undefined) {
      throw new Error(
        `Missing parameter '${param.name}' in ${tomlPath}. ` +
          `Expected parameters: ${metadata.parameters.map((p) => p.name).join(', ')}`
      );
    }

    // Flatten the value to strings and convert to field elements
    const strings = flattenValue(value);

    if (strings.length !== param.elementCount) {
      throw new Error(
        `Parameter '${param.name}' has ${strings.length} elements, ` +
          `but expected ${param.elementCount}`
      );
    }

    // Convert strings to field elements
    for (const str of strings) {
      allElements.push(parseFieldElement(str));
    }
  }

  return allElements;
}

/**
 * Load program metadata from JSON file
 */
export function loadMetadata(metadataPath: string): ProgramMetadata {
  const content = fs.readFileSync(metadataPath, 'utf-8');
  const data = JSON.parse(content) as {
    witnessCount: number;
    constraintCount: number;
    parameters: { name: string; elementCount: number }[];
  };

  return {
    witnessCount: data.witnessCount,
    constraintCount: data.constraintCount,
    parameters: data.parameters,
  };
}

/**
 * Get the default metadata path for a WASM file
 */
export function getMetadataPath(wasmPath: string): string {
  // output.wasm -> output.wasm.meta.json
  return `${wasmPath}.meta.json`;
}

/**
 * Parse inputs from Prover.toml using the metadata from the WASM file
 */
export function parseInputs(
  wasmPath: string,
  proverTomlPath: string
): { inputs: FieldElement[]; metadata: ProgramMetadata } {
  const metadataPath = getMetadataPath(wasmPath);
  const metadata = loadMetadata(metadataPath);
  const inputs = parseProverToml(proverTomlPath, metadata);
  return { inputs, metadata };
}
