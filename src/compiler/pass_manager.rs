use std::{fs, path::PathBuf};

use crate::compiler::{
    analysis::{
        instrumenter::{self, CostEstimator},
        types::{TypeInfo, Types},
        value_definitions::ValueDefinitions,
    },
    flow_analysis::FlowAnalysis,
    ssa::{DefaultSsaAnnotator, SSA},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataPoint {
    CFG,
    Types,
    ConstraintInstrumentation,
    ValueDefinitions,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PassInfo {
    pub name: &'static str,
    pub needs: Vec<DataPoint>,
}

pub trait Pass {
    fn run(&self, ssa: &mut SSA, pass_manager: &PassManager);
    fn pass_info(&self) -> PassInfo;
    fn invalidates_cfg(&self) -> bool {
        true
    }
    fn invalidates_types(&self) -> bool {
        true
    }
    fn invalidates_constraint_instrumentation(&self) -> bool {
        true
    }
    fn invalidates_value_definitions(&self) -> bool {
        true
    }
}

pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    current_pass_info: Option<PassInfo>,
    cfg: Option<FlowAnalysis>,
    draw_cfg: bool,
    type_info: Option<TypeInfo>,
    constraint_instrumentation: Option<instrumenter::Summary>,
    value_definitions: Option<ValueDefinitions>,
    debug_output_dir: Option<PathBuf>,
    phase_label: String,
}

impl PassManager {
    pub fn new(phase_label: String, draw_cfg: bool, passes: Vec<Box<dyn Pass>>) -> Self {
        Self {
            passes,
            current_pass_info: None,
            cfg: None,
            draw_cfg,
            type_info: None,
            constraint_instrumentation: None,
            value_definitions: None,
            debug_output_dir: None,
            phase_label,
        }
    }

    pub fn set_debug_output_dir(&mut self, debug_output_dir: PathBuf) {
        let specific_dir = debug_output_dir.join(self.phase_label.clone());
        if !specific_dir.exists() {
            fs::create_dir(&specific_dir).unwrap();
        }
        self.debug_output_dir = Some(specific_dir);
    }

    #[tracing::instrument(skip_all, name = "PassManager::run", fields(phase = %self.phase_label))]
    pub fn run(&mut self, ssa: &mut SSA) {
        if let Some(debug_output_dir) = &self.debug_output_dir {
            if debug_output_dir.exists() {
                fs::remove_dir_all(&debug_output_dir).unwrap();
            }
            fs::create_dir(&debug_output_dir).unwrap();
        }

        let passes = std::mem::take(&mut self.passes);
        for (i, pass) in passes.iter().enumerate() {
            self.run_pass(ssa, pass.as_ref(), i);
        }
        self.passes = passes;
        self.output_final_debug_info(ssa);
    }

    #[tracing::instrument(skip_all, fields(pass = %pass.pass_info().name))]
    fn run_pass(&mut self, ssa: &mut SSA, pass: &dyn Pass, pass_index: usize) {
        self.initialize_pass_data(ssa, &pass.pass_info());
        self.output_debug_info(ssa, pass_index, &pass.pass_info());
        self.current_pass_info = Some(pass.pass_info());
        pass.run(ssa, self);
        self.tear_down_pass_data(pass);
    }

    fn output_debug_info(&mut self, ssa: &SSA, pass_index: usize, pass_info: &PassInfo) {
        let Some(debug_output_dir) = &self.debug_output_dir else {
            return;
        };
        if pass_info.needs.contains(&DataPoint::CFG) && self.draw_cfg {
            if let Some(cfg) = &self.cfg {
                cfg.generate_images(
                    debug_output_dir.join(format!("before_pass_{}_{}", pass_index, pass_info.name)),
                    ssa,
                    format!("before {}: {}", pass_index, pass_info.name),
                );
                fs::write(
                    debug_output_dir
                        .join(format!("before_pass_{}_{}", pass_index, pass_info.name))
                        .join("code.txt"),
                    format!("{}", ssa.to_string(&DefaultSsaAnnotator)),
                )
                .unwrap();
            }
        }
    }

    fn output_final_debug_info(&mut self, ssa: &mut SSA) {
        if self.cfg.is_none() {
            self.cfg = Some(FlowAnalysis::run(ssa));
        }
        let Some(debug_output_dir) = &self.debug_output_dir else {
            return;
        };
        if self.draw_cfg {
            if let Some(cfg) = &self.cfg {
                cfg.generate_images(
                    debug_output_dir.join("final_result"),
                    ssa,
                    "final result".to_string(),
                );
            }
            fs::write(
                debug_output_dir.join("final_result").join("code.txt"),
                format!("{}", ssa.to_string(&DefaultSsaAnnotator)),
            )
            .unwrap();
        }
    }

    fn initialize_pass_data(&mut self, ssa: &mut SSA, pass_info: &PassInfo) {
        if (pass_info.needs.contains(&DataPoint::CFG)
            || pass_info.needs.contains(&DataPoint::Types)
            || pass_info
                .needs
                .contains(&DataPoint::ConstraintInstrumentation))
            && self.cfg.is_none()
        {
            self.cfg = Some(FlowAnalysis::run(ssa));
        }
        if (pass_info.needs.contains(&DataPoint::Types)
            || pass_info
                .needs
                .contains(&DataPoint::ConstraintInstrumentation))
            && self.type_info.is_none()
        {
            self.type_info = Some(Types::new().run(ssa, self.cfg.as_ref().unwrap()));
        }
        if pass_info
            .needs
            .contains(&DataPoint::ConstraintInstrumentation)
        {
            let cost_estimator = CostEstimator::new();
            let cost_analysis = cost_estimator.run(ssa, self.type_info.as_ref().unwrap());
            let summary = cost_analysis.summarize();
            self.constraint_instrumentation = Some(summary);
        }
        if pass_info.needs.contains(&DataPoint::ValueDefinitions) {
            self.value_definitions = Some(ValueDefinitions::from_ssa(ssa));
        }
    }

    fn tear_down_pass_data(&mut self, pass: &dyn Pass) {
        if pass.invalidates_cfg() {
            self.cfg = None;
        }
        if pass.invalidates_types() {
            self.type_info = None;
        }
        if pass.invalidates_constraint_instrumentation() {
            self.constraint_instrumentation = None;
        }
        if pass.invalidates_value_definitions() {
            self.value_definitions = None;
        }
    }

    pub fn get_cfg(&self) -> &FlowAnalysis {
        match &self.current_pass_info {
            Some(pass_info) => {
                if !pass_info.needs.contains(&DataPoint::CFG) {
                    panic!(
                        "Pass {} does not need CFG but tries to access it",
                        pass_info.name
                    );
                }
            }
            None => {}
        }
        self.cfg.as_ref().unwrap()
    }

    pub fn get_constraint_instrumentation(&self) -> &instrumenter::Summary {
        match &self.current_pass_info {
            Some(pass_info) => {
                if !pass_info
                    .needs
                    .contains(&DataPoint::ConstraintInstrumentation)
                {
                    panic!(
                        "Pass {} does not need constraint instrumentation but tries to access it",
                        pass_info.name
                    );
                }
            }
            None => {}
        }
        self.constraint_instrumentation.as_ref().unwrap()
    }

    pub fn get_type_info(&self) -> &TypeInfo {
        match &self.current_pass_info {
            Some(pass_info) => {
                if !pass_info.needs.contains(&DataPoint::Types) {
                    panic!(
                        "Pass {} does not need type information but tries to access it",
                        pass_info.name
                    );
                }
            }
            None => {}
        }
        self.type_info.as_ref().unwrap()
    }

    pub fn get_value_definitions(&self) -> &ValueDefinitions {
        match &self.current_pass_info {
            Some(pass_info) => {
                if !pass_info.needs.contains(&DataPoint::ValueDefinitions) {
                    panic!(
                        "Pass {} does not need value definitions but tries to access it",
                        pass_info.name
                    );
                }
            }
            None => {}
        }
        self.value_definitions.as_ref().unwrap()
    }
}
