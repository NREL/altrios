use super::*;
use crate::si;

#[test]
/// Unit test for standalone consist.
fn test_consist() {
    let mut consist = Consist::default();

    assert_eq!(consist.state.pwr_out_max, si::Power::ZERO);
    assert_eq!(consist.state.pwr_rate_out_max, si::PowerRate::ZERO);
    assert_eq!(consist.state.pwr_regen_max, si::Power::ZERO);
    consist
        .set_cur_pwr_max_out(None, todo!(), todo!(), 1.0 * uc::S)
        .unwrap();
    assert!(consist.state.pwr_out_max > si::Power::ZERO);
    assert!(consist.state.pwr_rate_out_max > si::PowerRate::ZERO);
    assert!(consist.state.pwr_regen_max == si::Power::ZERO);

    assert_eq!(consist.state.energy_out, si::Energy::ZERO);
    assert_eq!(consist.state.energy_fuel, si::Energy::ZERO);
    assert_eq!(consist.state.energy_res, si::Energy::ZERO);
    consist
        .solve_energy_consumption(uc::W * 1e6, todo!(), todo!(), uc::S * 1.0, Some(true))
        .unwrap();
    assert!(consist.state.energy_out > si::Energy::ZERO);
    assert!(consist.state.energy_fuel > si::Energy::ZERO);
    assert!(consist.state.energy_res > si::Energy::ZERO);
}
