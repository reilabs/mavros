use std::path::Path;

use mavros_vm::bytecode::{AllocationInstrumenter, AllocationType, AlocationEvent};
use plotters::prelude::*;

/// Computes the memory timeline from allocation events, draws a PNG chart to
/// `path`, and returns the final (residual) memory usage in bytes.
pub fn plot_memory_chart(instrumenter: &AllocationInstrumenter, path: &Path) -> usize {
    let mut stack_usage = Vec::new();
    let mut heap_usage = Vec::new();
    let mut current_stack = 0usize;
    let mut current_heap = 0usize;

    for event in &instrumenter.events {
        match event {
            AlocationEvent::Alloc(AllocationType::Stack, size) => {
                current_stack += size * 8;
            }
            AlocationEvent::Alloc(AllocationType::Heap, size) => {
                current_heap += size * 8;
            }
            AlocationEvent::Free(AllocationType::Stack, size) => {
                current_stack = current_stack.saturating_sub(*size * 8);
            }
            AlocationEvent::Free(AllocationType::Heap, size) => {
                current_heap = current_heap.saturating_sub(*size * 8);
            }
        }

        stack_usage.push(current_stack);
        heap_usage.push(current_heap);
    }

    if stack_usage.is_empty() {
        return 0;
    }

    draw_chart(path, &stack_usage, &heap_usage);

    *stack_usage.last().unwrap() + *heap_usage.last().unwrap()
}

fn draw_chart(path: &Path, stack_usage: &[usize], heap_usage: &[usize]) {
    let total_usage: Vec<usize> = stack_usage
        .iter()
        .zip(heap_usage.iter())
        .map(|(s, h)| s + h)
        .collect();

    let max_stack = *stack_usage.iter().max().unwrap_or(&1);
    let max_heap = *heap_usage.iter().max().unwrap_or(&1);
    let max_total = *total_usage.iter().max().unwrap_or(&1);

    let root = BitMapBackend::new(path, (2400, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let (left, rest) = root.split_horizontally(800);
    let (middle, right) = rest.split_horizontally(800);

    let common_max = max_total.max(max_stack).max(max_heap);

    let (_unit, divisor, y_label) = if common_max >= 2 * 1024 * 1024 {
        ("MB", 1024 * 1024, "Memory Size (MB)".to_string())
    } else if common_max >= 2 * 1024 {
        ("KB", 1024, "Memory Size (KB)".to_string())
    } else {
        ("B", 1, "Memory Size (bytes)".to_string())
    };

    let total_data: Vec<(usize, f64)> = total_usage
        .iter()
        .enumerate()
        .map(|(i, &size)| (i, size as f64 / divisor as f64))
        .collect();

    let stack_data: Vec<(usize, f64)> = stack_usage
        .iter()
        .enumerate()
        .map(|(i, &size)| (i, size as f64 / divisor as f64))
        .collect();

    let heap_data: Vec<(usize, f64)> = heap_usage
        .iter()
        .enumerate()
        .map(|(i, &size)| (i, size as f64 / divisor as f64))
        .collect();

    let y_max = common_max as f64 / divisor as f64;

    let mut chart1 = ChartBuilder::on(&left)
        .caption("Total Memory Usage", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..total_usage.len(), 0.0..y_max)
        .unwrap();

    chart1
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .x_desc("Event Number")
        .y_desc(y_label.clone())
        .draw()
        .unwrap();

    chart1
        .draw_series(
            total_data
                .iter()
                .map(|&(x, y)| Rectangle::new([(x, 0.0), (x + 1, y)], GREEN.filled())),
        )
        .unwrap()
        .label("Total Memory")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    chart1
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    let mut chart2 = ChartBuilder::on(&middle)
        .caption("Stack Memory Usage", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..stack_usage.len(), 0.0..y_max)
        .unwrap();

    chart2
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .x_desc("Event Number")
        .y_desc(y_label.clone())
        .draw()
        .unwrap();

    chart2
        .draw_series(
            stack_data
                .iter()
                .map(|&(x, y)| Rectangle::new([(x, 0.0), (x + 1, y)], BLUE.filled())),
        )
        .unwrap()
        .label("Stack Memory")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart2
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    let mut chart3 = ChartBuilder::on(&right)
        .caption("Heap Memory Usage", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..heap_usage.len(), 0.0..y_max)
        .unwrap();

    chart3
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .x_desc("Event Number")
        .y_desc(y_label.clone())
        .draw()
        .unwrap();

    chart3
        .draw_series(
            heap_data
                .iter()
                .map(|&(x, y)| Rectangle::new([(x, 0.0), (x + 1, y)], RED.filled())),
        )
        .unwrap()
        .label("Heap Memory")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart3
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
}
