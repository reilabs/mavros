//! Utilities for external WebAssembly DWARF sidecars.

const WASM_HEADER: &[u8; 8] = b"\0asm\x01\0\0\0";
const EXTERNAL_DEBUG_INFO: &str = "external_debug_info";

struct Section<'a> {
    id: u8,
    bytes: &'a [u8],
    custom_name: Option<&'a str>,
    custom_data: Option<&'a [u8]>,
}

fn read_u32(bytes: &[u8], mut offset: usize) -> Result<(u32, usize), String> {
    let mut value = 0u32;
    for shift in (0..35).step_by(7) {
        let byte = *bytes
            .get(offset)
            .ok_or_else(|| "truncated unsigned LEB128 value".to_string())?;
        offset += 1;
        value |= u32::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, offset));
        }
    }
    Err("invalid unsigned LEB128 value".to_string())
}

fn write_u32(mut value: u32, output: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn sections(bytes: &[u8]) -> Result<Vec<Section<'_>>, String> {
    if !bytes.starts_with(WASM_HEADER) {
        return Err("invalid WebAssembly header".to_string());
    }

    let mut result = Vec::new();
    let mut offset = WASM_HEADER.len();
    while offset < bytes.len() {
        let section_start = offset;
        let id = bytes[offset];
        offset += 1;
        let (payload_len, payload_start) = read_u32(bytes, offset)?;
        let section_end = payload_start
            .checked_add(payload_len as usize)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| "section extends beyond end of WebAssembly module".to_string())?;

        let (custom_name, custom_data) = if id == 0 {
            let (name_len, name_start) = read_u32(bytes, payload_start)?;
            let name_end = name_start
                .checked_add(name_len as usize)
                .filter(|end| *end <= section_end)
                .ok_or_else(|| "custom-section name extends beyond its payload".to_string())?;
            let name = std::str::from_utf8(&bytes[name_start..name_end])
                .map_err(|_| "custom-section name is not UTF-8".to_string())?;
            (Some(name), Some(&bytes[name_end..section_end]))
        } else {
            (None, None)
        };

        result.push(Section {
            id,
            bytes: &bytes[section_start..section_end],
            custom_name,
            custom_data,
        });
        offset = section_end;
    }
    Ok(result)
}

fn is_dwarf_section(name: &str) -> bool {
    name.starts_with(".debug_") || name.starts_with(".zdebug_") || name.starts_with("reloc..debug_")
}

fn append_custom_section(output: &mut Vec<u8>, name: &str, contents: &[u8]) {
    output.push(0);
    let name_len = u32::try_from(name.len()).expect("WASM custom-section name is too large");
    let mut payload = Vec::new();
    write_u32(name_len, &mut payload);
    payload.extend_from_slice(name.as_bytes());
    payload.extend_from_slice(contents);
    let payload_len = u32::try_from(payload.len()).expect("WASM custom section is too large");
    write_u32(payload_len, output);
    output.extend_from_slice(&payload);
}

/// Strip DWARF from an executable module, add its sidecar URL, and retain the original linked
/// module as the standalone debug companion. Keeping the module structure in the companion is
/// compatible with browser DWARF tooling that needs type, function, and code sections.
pub(crate) fn split_debug_info(
    bytes: &[u8],
    external_url: &str,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut executable = WASM_HEADER.to_vec();
    let mut found_dwarf = false;
    for section in sections(bytes)? {
        let name = section.custom_name.unwrap_or_default();
        if is_dwarf_section(name) {
            found_dwarf = true;
            continue;
        }
        // Replace a stale association rather than emitting duplicate URLs.
        if section.id == 0 && name == EXTERNAL_DEBUG_INFO {
            continue;
        }
        executable.extend_from_slice(section.bytes);
    }
    if !found_dwarf {
        return Err("linked WebAssembly module contains no DWARF sections".to_string());
    }
    append_custom_section(
        &mut executable,
        EXTERNAL_DEBUG_INFO,
        external_url.as_bytes(),
    );
    Ok((executable, bytes.to_vec()))
}

/// Read the relative URL stored in a module's `external_debug_info` custom section.
pub fn external_debug_info_url(bytes: &[u8]) -> Result<Option<String>, String> {
    for section in sections(bytes)? {
        if section.custom_name == Some(EXTERNAL_DEBUG_INFO) {
            let url = std::str::from_utf8(section.custom_data.unwrap_or_default())
                .map_err(|_| "external_debug_info URL is not UTF-8".to_string())?;
            return Ok(Some(url.to_string()));
        }
    }
    Ok(None)
}

/// Combine a stripped executable and its debug companion in memory for runtimes such as Wasmtime
/// that consume embedded DWARF but do not follow `external_debug_info` themselves.
pub fn merge_debug_info(executable: &[u8], debug: &[u8]) -> Result<Vec<u8>, String> {
    let mut merged = WASM_HEADER.to_vec();
    for section in sections(executable)? {
        if !section.custom_name.is_some_and(is_dwarf_section) {
            merged.extend_from_slice(section.bytes);
        }
    }

    let mut found_dwarf = false;
    for section in sections(debug)? {
        if section.custom_name.is_some_and(is_dwarf_section) {
            found_dwarf = true;
            merged.extend_from_slice(section.bytes);
        }
    }
    if !found_dwarf {
        return Err("debug companion contains no DWARF sections".to_string());
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn custom_section(name: &str, contents: &[u8]) -> Vec<u8> {
        let mut section = WASM_HEADER.to_vec();
        append_custom_section(&mut section, name, contents);
        section.drain(..WASM_HEADER.len());
        section
    }

    #[test]
    fn split_module_links_to_full_debug_companion() {
        let mut module = WASM_HEADER.to_vec();
        module.extend_from_slice(&[1, 1, 0]);
        module.extend(custom_section("name", &[1, 2]));
        module.extend(custom_section(".debug_info", &[3, 4, 5]));
        module.extend(custom_section(".debug_line", &[6, 7]));

        let (executable, debug) = split_debug_info(&module, "program.debug.wasm").unwrap();

        assert_eq!(
            external_debug_info_url(&executable).unwrap().as_deref(),
            Some("program.debug.wasm")
        );
        assert!(
            !executable
                .windows(11)
                .any(|window| window == b".debug_info")
        );
        assert!(debug.windows(11).any(|window| window == b".debug_info"));
        assert!(debug.windows(4).any(|window| window == b"name"));
        assert!(debug.windows(3).any(|window| window == [1, 1, 0]));
    }

    #[test]
    fn merged_runtime_module_restores_dwarf_without_duplicate_code() {
        let mut module = WASM_HEADER.to_vec();
        module.extend_from_slice(&[1, 1, 0]);
        module.extend(custom_section(".debug_info", &[3, 4, 5]));
        let (executable, debug) = split_debug_info(&module, "program.debug.wasm").unwrap();

        let merged = merge_debug_info(&executable, &debug).unwrap();

        assert_eq!(
            merged
                .windows(3)
                .filter(|window| *window == [1, 1, 0])
                .count(),
            1
        );
        assert_eq!(
            merged
                .windows(11)
                .filter(|window| *window == b".debug_info")
                .count(),
            1
        );
    }
}
