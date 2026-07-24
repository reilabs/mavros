use std::{
    collections::BTreeMap,
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use mavros_artifacts::FlamegraphProfile;

const SYNTHETIC_PROFILE_START_TIME: u64 = 1_000_000;

#[derive(Debug)]
struct CpuProfileNode {
    id: u64,
    function_name: String,
    hit_count: u64,
    children: Vec<u64>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SerializedCpuProfile {
    nodes: Vec<SerializedCpuProfileNode>,
    start_time: u64,
    end_time: u64,
    samples: Vec<u64>,
    time_deltas: Vec<u64>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SerializedCpuProfileNode {
    id: u64,
    call_frame: CpuProfileCallFrame,
    hit_count: u64,
    children: Vec<u64>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct CpuProfileCallFrame {
    function_name: String,
    script_id: &'static str,
    url: &'static str,
    line_number: i64,
    column_number: i64,
}

/// Write a folded profile and, when available, render its interactive SVG with
/// Brendan Gregg's `flamegraph.pl` from `PATH`.
pub fn render(
    profile: &FlamegraphProfile,
    output_dir: &Path,
    name: &str,
    title: &str,
    count_name: &str,
) -> Result<(PathBuf, Option<PathBuf>), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;
    let folded_path = output_dir.join(format!("{name}.folded"));
    let svg_path = output_dir.join(format!("{name}.svg"));
    let folded = profile.to_folded();
    fs::write(&folded_path, &folded)?;
    let render_input = if profile.is_empty() {
        "<no samples> 1\n"
    } else {
        folded.as_str()
    };
    let render_title = if profile.is_empty() {
        format!("{title} (no samples)")
    } else {
        title.to_string()
    };
    let render_count_name = if profile.is_empty() {
        "placeholder"
    } else {
        count_name
    };

    let child = Command::new("flamegraph.pl")
        .args([
            "--hash",
            "--title",
            &render_title,
            "--countname",
            render_count_name,
            "--nametype",
            "Function:",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok((folded_path, None));
        }
        Err(error) => {
            return Err(io::Error::new(
                error.kind(),
                format!("failed to start flamegraph.pl: {error}"),
            )
            .into());
        }
    };

    child
        .stdin
        .take()
        .expect("piped FlameGraph stdin is available")
        .write_all(render_input.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "flamegraph.pl failed for {}: {}",
            folded_path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
        .into());
    }
    fs::write(&svg_path, output.stdout)?;
    Ok((folded_path, Some(svg_path)))
}

/// Export an ordered deterministic instruction profile in the Chrome DevTools
/// CPU profile format. Large profiles use deterministic stratified instruction
/// sampling in [`FlamegraphProfile`] to keep recording memory and output size
/// bounded without synchronizing to periodic workloads.
pub fn write_cpuprofile(
    profile: &FlamegraphProfile,
    output_dir: &Path,
    name: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;
    let output_path = output_dir.join(format!("{name}.cpuprofile"));
    fs::write(&output_path, cpuprofile_json(profile)?)?;
    Ok(output_path)
}

fn cpuprofile_json(profile: &FlamegraphProfile) -> Result<String, serde_json::Error> {
    serde_json::to_string(&cpuprofile(profile))
}

fn cpuprofile(profile: &FlamegraphProfile) -> SerializedCpuProfile {
    let mut nodes = vec![CpuProfileNode {
        id: 1,
        function_name: "(root)".to_string(),
        hit_count: 0,
        children: Vec::new(),
    }];
    let mut node_by_parent_and_name = BTreeMap::<(u64, String), u64>::new();
    let mut leaf_by_stack = BTreeMap::<Vec<String>, u64>::new();

    for (stack, _) in profile.stacks() {
        let mut parent_id = 1;
        for function_name in stack {
            let key = (parent_id, function_name.clone());
            let node_id = if let Some(node_id) = node_by_parent_and_name.get(&key) {
                *node_id
            } else {
                let node_id = nodes.len() as u64 + 1;
                nodes[(parent_id - 1) as usize].children.push(node_id);
                nodes.push(CpuProfileNode {
                    id: node_id,
                    function_name: function_name.clone(),
                    hit_count: 0,
                    children: Vec::new(),
                });
                node_by_parent_and_name.insert(key, node_id);
                node_id
            };
            parent_id = node_id;
        }
        leaf_by_stack.insert(stack.to_vec(), parent_id);
    }

    let mut samples = Vec::new();
    let mut time_deltas = Vec::new();
    let mut previous_timestamp = 0;
    for (position, stack) in profile.timeline_samples() {
        let node_id = leaf_by_stack[stack];
        nodes[(node_id - 1) as usize].hit_count += 1;
        samples.push(node_id);
        let timestamp = position + 1;
        time_deltas.push(timestamp - previous_timestamp);
        previous_timestamp = timestamp;
    }

    let nodes = nodes
        .into_iter()
        .map(|node| SerializedCpuProfileNode {
            id: node.id,
            call_frame: CpuProfileCallFrame {
                function_name: node.function_name,
                script_id: "0",
                url: "",
                line_number: -1,
                column_number: -1,
            },
            hit_count: node.hit_count,
            children: node.children,
        })
        .collect::<Vec<_>>();

    SerializedCpuProfile {
        nodes,
        start_time: SYNTHETIC_PROFILE_START_TIME,
        end_time: SYNTHETIC_PROFILE_START_TIME + profile.total_weight(),
        samples,
        time_deltas,
    }
}

#[cfg(test)]
mod tests {
    use super::{SYNTHETIC_PROFILE_START_TIME, cpuprofile, cpuprofile_json};
    use mavros_artifacts::FlamegraphProfile;

    fn cpuprofile_value(profile: &FlamegraphProfile) -> serde_json::Value {
        serde_json::to_value(cpuprofile(profile)).unwrap()
    }

    #[test]
    fn cpuprofile_preserves_call_paths_and_instruction_weights() {
        let mut profile = FlamegraphProfile::default();
        profile.record(["main"], 3);
        profile.record(["main", "helper"], 7);
        profile.record(["main", "other"], 2);

        let value = cpuprofile_value(&profile);
        assert_eq!(value["startTime"], SYNTHETIC_PROFILE_START_TIME);
        assert_eq!(value["endTime"], SYNTHETIC_PROFILE_START_TIME + 12);
        assert_eq!(
            value["samples"],
            serde_json::json!([2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4])
        );
        assert_eq!(value["timeDeltas"], serde_json::json!(vec![1; 12]));
        assert_eq!(value["nodes"][0]["children"], serde_json::json!([2]));
        assert_eq!(value["nodes"][1]["children"], serde_json::json!([3, 4]));
        assert_eq!(value["nodes"][2]["callFrame"]["functionName"], "helper");
        assert_eq!(value["nodes"][2]["hitCount"], 7);
    }

    #[test]
    fn cpuprofile_preserves_execution_order() {
        let mut profile = FlamegraphProfile::default();
        profile.record(["main"], 2);
        profile.record(["main", "helper"], 2);
        profile.record(["main"], 2);

        let value = cpuprofile_value(&profile);
        assert_eq!(value["samples"], serde_json::json!([2, 2, 3, 3, 2, 2]));
        assert_eq!(value["timeDeltas"], serde_json::json!(vec![1; 6]));
    }

    #[test]
    fn cpuprofile_starts_with_the_signature_expected_by_chrome() {
        let mut profile = FlamegraphProfile::default();
        profile.record(["main"], 1);

        let json = cpuprofile_json(&profile).unwrap();
        assert!(json.starts_with("{\"nodes\":["));
    }

    #[test]
    fn cpuprofile_downsamples_large_instruction_profiles() {
        let mut profile = FlamegraphProfile::default();
        profile.record(["main"], 250_001);

        let value = cpuprofile_value(&profile);
        let samples = value["samples"].as_array().unwrap();
        let time_deltas = value["timeDeltas"].as_array().unwrap();
        assert!((62_499..=62_501).contains(&samples.len()));
        assert_eq!(time_deltas.len(), samples.len());
        assert!(time_deltas.iter().all(|delta| delta.as_u64().unwrap() > 0));
        assert_eq!(value["endTime"], SYNTHETIC_PROFILE_START_TIME + 250_001);
        assert_eq!(
            value["nodes"][1]["hitCount"].as_u64().unwrap() as usize,
            samples.len()
        );
    }
}
