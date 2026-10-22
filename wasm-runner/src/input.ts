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
  hasReturn: boolean;
  returnElementCount: number;
  entryBlobFieldCount: number;
  // Where the computed return sits in the witness, when the program has one.
  returnColumns?: { columnStart: number | null; len: number };
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

function readProverToml(tomlPath: string): Record<string, unknown> {
  const content = fs.readFileSync(tomlPath, 'utf-8');
  return TOML.parse(content) as Record<string, unknown>;
}

function entryBlobFromParsed(
  parsed: Record<string, unknown>,
  tomlPath: string,
  metadata: ProgramMetadata
): FieldElement[] {
  const allElements: FieldElement[] = [];

  for (const param of metadata.parameters) {
    const value = parsed[param.name];

    if (value === undefined) {
      throw new Error(
        `Missing parameter '${param.name}' in ${tomlPath}. ` +
          `Expected parameters: ${metadata.parameters.map((p) => p.name).join(', ')}`
      );
    }

    const strings = flattenValue(value);

    if (strings.length !== param.elementCount) {
      throw new Error(
        `Parameter '${param.name}' has ${strings.length} elements, ` +
          `but expected ${param.elementCount}`
      );
    }

    for (const str of strings) {
      allElements.push(parseFieldElement(str));
    }
  }

  if (allElements.length !== metadata.entryBlobFieldCount) {
    throw new Error(
      `Assembled ${allElements.length} input elements, ` +
        `but the program expects ${metadata.entryBlobFieldCount}`
    );
  }

  return allElements;
}

function declaredReturnFromParsed(
  parsed: Record<string, unknown>,
  metadata: ProgramMetadata
): FieldElement[] | null {
  if (!metadata.hasReturn) {
    return null;
  }
  const returnValue = parsed['return'];
  if (returnValue === undefined) {
    console.warn(
      'Warning: the program declares a return value but Prover.toml does not ' +
        'supply `return`; the computed return is not checked'
    );
    return null;
  }
  const strings = flattenValue(returnValue);
  if (strings.length !== metadata.returnElementCount) {
    throw new Error(
      `Return value has ${strings.length} elements, ` +
        `but expected ${metadata.returnElementCount}`
    );
  }
  return strings.map(parseFieldElement);
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
    hasReturn: boolean;
    returnElementCount: number;
    entryBlobFieldCount: number;
    witnessManifest?: { returns?: { columnStart: number | null; len: number } | null };
  };

  return {
    witnessCount: data.witnessCount,
    constraintCount: data.constraintCount,
    parameters: data.parameters,
    hasReturn: data.hasReturn,
    returnElementCount: data.returnElementCount,
    entryBlobFieldCount: data.entryBlobFieldCount,
    returnColumns: data.witnessManifest?.returns ?? undefined,
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
): {
  inputs: FieldElement[];
  declaredReturn: FieldElement[] | null;
  metadata: ProgramMetadata;
} {
  const metadataPath = getMetadataPath(wasmPath);
  const metadata = loadMetadata(metadataPath);
  const parsed = readProverToml(proverTomlPath);
  const inputs = entryBlobFromParsed(parsed, proverTomlPath, metadata);
  const declaredReturn = declaredReturnFromParsed(parsed, metadata);
  return { inputs, declaredReturn, metadata };
}
