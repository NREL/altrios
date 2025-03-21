use super::*;

/// Trait for ensuring consistency among locomotives and consists
pub trait LocoTrait {
    /// Sets current max power, current max power rate, and current max regen
    /// power that can be absorbed by the RES/battery
    ///
    /// # Arguments:
    /// - `pwr_aux`: aux power
    /// - `train_speed`: current train speed
    /// - `train_mass`: portion of total train mass handled by `self`
    /// - `dt`: time step size
    fn set_curr_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
        dt: si::Time,
    ) -> anyhow::Result<()>;
    /// Save current state
    fn save_state(&mut self) {
        unimplemented!();
    }
    /// Step counter
    fn step(&mut self) {
        unimplemented!();
    }
    /// Get energy loss in components
    fn get_energy_loss(&self) -> si::Energy;
}

#[altrios_api(
    #[new]
    /// Rust-defined `__new__` magic method for Python used exposed via PyO3.
    fn __new__(v: Vec<Locomotive>) -> Self {
        Self(v)
    }
)]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
/// Wrapper struct for `Vec<Locomotive>` to expose various methods to Python.
pub struct Pyo3VecLocoWrapper(pub Vec<Locomotive>);
impl Pyo3VecLocoWrapper {
    pub fn new(value: Vec<Locomotive>) -> Self {
        Self(value)
    }
}

impl Init for Pyo3VecLocoWrapper {
    fn init(&mut self) -> Result<(), Error> {
        self.0.iter_mut().try_for_each(|l| l.init())?;
        Ok(())
    }
}
impl SerdeAPI for Pyo3VecLocoWrapper {}

pub trait SolvePower {
    /// Returns vector of locomotive tractive powers during positive traction events
    fn solve_positive_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<Vec<si::Power>>;
    fn solve_negative_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<Vec<si::Power>>;
}

#[derive(PartialEq, Eq, Clone, Deserialize, Serialize, Debug, SerdeAPI)]
/// Similar to [self::Proportional], but positive traction conditions use locomotives with
/// ReversibleEnergyStorage preferentially, within their power limits.  Recharge is same as
/// `Proportional` variant.
pub struct RESGreedy;
impl SolvePower for RESGreedy {
    fn solve_positive_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
        _train_mass: Option<si::Mass>,
        _train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<Vec<si::Power>> {
        let loco_pwr_out_vec: Vec<si::Power> = if state.pwr_out_deficit == si::Power::ZERO {
            // draw all power from RES-equipped locomotives
            loco_vec
                .iter()
                .map(|loco| match &loco.loco_type {
                    PowertrainType::ConventionalLoco(_) => si::Power::ZERO,
                    PowertrainType::HybridLoco(_) => {
                        loco.state.pwr_out_max / state.pwr_out_max_reves * state.pwr_out_req
                    }
                    PowertrainType::BatteryElectricLoco(_) => {
                        loco.state.pwr_out_max / state.pwr_out_max_reves * state.pwr_out_req
                    }
                    // if the DummyLoco is present in the consist, it should be the only locomotive
                    // and pwr_out_deficit should be 0.0
                    PowertrainType::DummyLoco(_) => state.pwr_out_req,
                })
                .collect()
        } else {
            // draw deficit power from conventional and hybrid locomotives
            loco_vec
                .iter()
                .map(|loco| match &loco.loco_type {
                    PowertrainType::ConventionalLoco(_) => {
                        loco.state.pwr_out_max / state.pwr_out_max_non_reves
                            * state.pwr_out_deficit
                    }
                    PowertrainType::HybridLoco(_) => loco.state.pwr_out_max,
                    PowertrainType::BatteryElectricLoco(_) => loco.state.pwr_out_max,
                    PowertrainType::DummyLoco(_) => {
                        si::Power::ZERO /* this else branch should not happen when DummyLoco is present */
                    }
                })
                .collect()
        };
        let loco_pwr_out_vec_sum: si::Power = loco_pwr_out_vec.iter().copied().sum();
        ensure!(
            utils::almost_eq_uom(
                &loco_pwr_out_vec.iter().copied().sum(),
                &state.pwr_out_req,
                None,
            ),
            format!(
                "{}\n{}",
                format_dbg!(loco_pwr_out_vec_sum.get::<si::kilowatt>()),
                format_dbg!(&state.pwr_out_req.get::<si::kilowatt>())
            )
        );
        Ok(loco_pwr_out_vec)
    }

    fn solve_negative_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<Vec<si::Power>> {
        solve_negative_traction(loco_vec, state, train_mass, train_speed)
    }
}

fn get_pwr_regen_vec(loco_vec: &[Locomotive], regen_frac: si::Ratio) -> Vec<si::Power> {
    loco_vec
        .iter()
        .map(|loco| match &loco.loco_type {
            // no braking power from conventional locos if there is capacity to regen all power
            PowertrainType::ConventionalLoco(_) => si::Power::ZERO,
            PowertrainType::HybridLoco(_) => loco.state.pwr_regen_max * regen_frac,
            PowertrainType::BatteryElectricLoco(_) => loco.state.pwr_regen_max * regen_frac,
            // if the DummyLoco is present in the consist, it should be the only locomotive
            // and pwr_regen_deficit should be 0.0
            PowertrainType::DummyLoco(_) => si::Power::ZERO,
        })
        .collect()
}

/// Used for apportioning negative tractive power throughout consist for several
/// [PowerDistributionControlType] variants
fn solve_negative_traction(
    loco_vec: &[Locomotive],
    consist_state: &ConsistState,
    _train_mass: Option<si::Mass>,
    _train_speed: Option<si::Velocity>,
) -> anyhow::Result<Vec<si::Power>> {
    // positive during any kind of negative traction event
    let pwr_brake_req = -consist_state.pwr_out_req;

    // fraction of consist-level max regen required to fulfill required braking power
    let regen_frac = if consist_state.pwr_regen_max == si::Power::ZERO {
        // divide-by-zero protection
        si::Ratio::ZERO
    } else {
        (pwr_brake_req / consist_state.pwr_regen_max).min(uc::R * 1.)
    };
    let pwr_out_vec: Vec<si::Power> = if consist_state.pwr_regen_deficit == si::Power::ZERO {
        get_pwr_regen_vec(loco_vec, regen_frac)
    } else {
        // In this block, we know that all of the regen capability will be used so the goal is to spread
        // dynamic braking effort among the non-RES-equipped and then all locomotives up until they're doing
        // the same dynmamic braking effort
        let pwr_regen_vec = get_pwr_regen_vec(loco_vec, regen_frac);
        // extra dynamic braking power after regen has been subtracted off
        let pwr_surplus_vec: Vec<si::Power> = loco_vec
            .iter()
            .zip(&pwr_regen_vec)
            // this `unwrap` might cause problems for DummyLoco
            .map(|(loco, pwr_regen)| loco.electric_drivetrain().unwrap().pwr_out_max - *pwr_regen)
            .collect();
        let pwr_surplus_sum = pwr_surplus_vec
            .iter()
            .fold(0.0 * uc::W, |acc, &curr| acc + curr);

        // needed braking power not including regen per total available braking power not including regen
        let surplus_frac = consist_state.pwr_regen_deficit / pwr_surplus_sum;
        ensure!(
            surplus_frac >= si::Ratio::ZERO && surplus_frac <= uc::R,
            format_dbg!(surplus_frac),
        );
        // total dynamic braking, including regen
        let pwr_dyn_brake_vec: Vec<si::Power> = pwr_surplus_vec
            .iter()
            .zip(pwr_regen_vec)
            .map(|(pwr_surplus, pwr_regen)| *pwr_surplus * surplus_frac + pwr_regen)
            .collect();
        pwr_dyn_brake_vec
    };
    // negate it to be consistent with sign convention
    let pwr_out_vec: Vec<si::Power> = pwr_out_vec.iter().map(|x| -*x).collect();
    Ok(pwr_out_vec)
}

#[derive(PartialEq, Eq, Clone, Deserialize, Serialize, Debug, SerdeAPI)]
/// During positive traction, power is proportional to each locomotive's current max
/// available power.  During negative traction, any power that's less negative than the total
/// sum of the regen capacity is distributed to each locomotive with regen capacity, proportionally
/// to it's current max regen ability.
pub struct Proportional;
#[allow(unused_variables)]
#[allow(unreachable_code)]
impl SolvePower for Proportional {
    fn solve_positive_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
        _train_mass: Option<si::Mass>,
        _train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<Vec<si::Power>> {
        todo!("this need some attention to make sure it handles the hybrid correctly");
        Ok(loco_vec
            .iter()
            .map(|loco| {
                // loco.state.pwr_out_max already accounts for rate
                loco.state.pwr_out_max / state.pwr_out_max * state.pwr_out_req
            })
            .collect())
    }

    fn solve_negative_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<Vec<si::Power>> {
        todo!("this need some attention to make sure it handles the hybrid correctly");
        solve_negative_traction(loco_vec, state, train_mass, train_speed)
    }
}

#[derive(PartialEq, Eq, Clone, Deserialize, Serialize, Debug)]
/// Control strategy for when locomotives are located at both the front and back of the train.
pub struct FrontAndBack;
impl SerdeAPI for FrontAndBack {}
impl Init for FrontAndBack {}
impl SolvePower for FrontAndBack {
    fn solve_positive_traction(
        &mut self,
        _loco_vec: &[Locomotive],
        _state: &ConsistState,
        _train_mass: Option<si::Mass>,
        _train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<Vec<si::Power>> {
        todo!() // not needed urgently
    }

    fn solve_negative_traction(
        &mut self,
        _loco_vec: &[Locomotive],
        _state: &ConsistState,
        _train_mass: Option<si::Mass>,
        _train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<Vec<si::Power>> {
        todo!() // not needed urgently
    }
}

/// Variants of this enum are used to determine what control strategy gets used for distributing
/// power required from or delivered to during negative tractive power each locomotive.
#[derive(PartialEq, Clone, Deserialize, Serialize, Debug, SerdeAPI)]
pub enum PowerDistributionControlType {
    RESGreedy(RESGreedy),
    Proportional(Proportional),
    FrontAndBack(FrontAndBack),
}

impl SolvePower for PowerDistributionControlType {
    fn solve_negative_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<Vec<si::Power>> {
        match self {
            Self::RESGreedy(res_greedy) => {
                res_greedy.solve_negative_traction(loco_vec, state, train_mass, train_speed)
            }
            Self::Proportional(prop) => {
                prop.solve_negative_traction(loco_vec, state, train_mass, train_speed)
            }
            Self::FrontAndBack(fab) => {
                fab.solve_negative_traction(loco_vec, state, train_mass, train_speed)
            }
        }
    }

    fn solve_positive_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<Vec<si::Power>> {
        match self {
            Self::RESGreedy(res_greedy) => {
                res_greedy.solve_positive_traction(loco_vec, state, train_mass, train_speed)
            }
            Self::Proportional(prop) => {
                prop.solve_positive_traction(loco_vec, state, train_mass, train_speed)
            }
            Self::FrontAndBack(fab) => {
                fab.solve_positive_traction(loco_vec, state, train_mass, train_speed)
            }
        }
    }
}

impl Default for PowerDistributionControlType {
    fn default() -> Self {
        Self::RESGreedy(RESGreedy)
    }
}
